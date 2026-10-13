// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/GpuVulkan.h"

#include "Luau/GpuShaderCompiler.h"

#include "lua.h"
#include "lualib.h"

#include <vulkan/vulkan.h>

#include <string.h>

#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Luau
{
namespace Gpu
{

namespace
{

// registry tables, weak on their keys, that map functions to kernels and buffers to the device copies of them
const char* const kKernelsKey = "_GPUKERNELS";
const char* const kBuffersKey = "_GPUBUFFERS";

struct KernelEntry
{
    std::string key;
    // functions using the kernel, in every state; it's freed with the last one
    int users = 0;

    std::string name;
    std::string reason;
    bool warned = false;
    std::vector<bool> written;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

// device memory for a Luau buffer and the host memory mirroring it
struct Allocation
{
    VkDeviceSize deviceSize = 0;

    VkBuffer device = VK_NULL_HANDLE;
    VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

// device copy of a Luau buffer
struct BufferEntry : Allocation
{
    size_t size = 0;

    // contents of a dirty copy before the dispatch that is running, to restore them if it fails
    VkBuffer backup = VK_NULL_HANDLE;
    VkDeviceMemory backupMemory = VK_NULL_HANDLE;

    // the device holds defined contents; when not dirty, they are those of the staging memory and the Luau buffer
    bool valid = false;
    // the device contents are newer than the Luau buffer's (written by a batch that didn't list the buffer as an output)
    bool dirty = false;
    uint64_t lastUsed = 0;

    // position in the least recently used list, which dirty copies leave since they can't be evicted
    bool listed = false;
    std::list<uint64_t>::iterator lru;
};

// lives in the registry table of buffers; once the Luau buffer is collected, so is this (in the next GC cycle, since weak
// tables mark their values), and the device copy is freed
struct BufferHandle
{
    VulkanBackend::Impl* impl;
    uint64_t id;
};

// the same for functions and their kernels
struct KernelHandle
{
    VulkanBackend::Impl* impl;
    KernelEntry* kernel;
};

// a buffer used by a dispatch, once however many times it's bound
struct Use
{
    BufferEntry* entry;
    void* data;
    bool written;
    bool output;
    bool changed;
};

struct Limits
{
    uint32_t storageBuffers;
    uint64_t storageRange;
    uint32_t groupsX;
    uint32_t groupsY;
    uint32_t pushConstants;
};

// limits every Vulkan implementation guarantees
constexpr Limits kMinimumLimits = {4, 1u << 27, 65535, 65535, 128};

void barrier(VkCommandBuffer cb, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
{
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = srcAccess;
    mb.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void copyBuffer(VkCommandBuffer cb, VkBuffer from, VkBuffer to, VkDeviceSize size)
{
    VkBufferCopy copy = {0, 0, size};
    vkCmdCopyBuffer(cb, from, to, 1, &copy);
}

// pushes the registry table that is weak on its keys, creating it the first time
void pushWeakTable(lua_State* L, const char* key)
{
    if (lua_rawgetfield(L, LUA_REGISTRYINDEX, key) != LUA_TNIL)
        return;

    lua_pop(L, 1);
    lua_newtable(L);
    lua_createtable(L, 0, 1);
    lua_pushstring(L, "k");
    lua_setfield(L, -2, "__mode");
    lua_setmetatable(L, -2);
    lua_pushvalue(L, -1);
    lua_rawsetfield(L, LUA_REGISTRYINDEX, key);
}

} // namespace

struct VulkanBackend::Impl
{
    VulkanOptions options;
    Limits limits = kMinimumLimits;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkPhysicalDeviceMemoryProperties memory = {};

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    uint32_t descriptorSets = 0;

    BufferEntry status;

    ShaderCompiler shaders;
    std::function<CodeGen::GpuKernel(lua_State*, int)> lowering = CodeGen::compileGpu;

    std::string deviceName;
    DispatchStats stats;
    uint64_t dispatchCounter = 0;

    // set when the device stopped responding or was lost; every dispatch then runs in the VM
    std::string disabledReason;
    bool deviceBusy = false;

    // kernels are shared by every function lowering to the same shader; rejected functions share one per reason
    std::unordered_map<std::string, std::unique_ptr<KernelEntry>> kernels;

    std::unordered_map<uint64_t, BufferEntry> buffers;
    std::list<uint64_t> lru;
    uint64_t nextBufferId = 0;
    uint64_t cachedBytes = 0;


    // buffers and functions collected by the GC; what they hold is freed on the next call into the backend, since the GC
    // may run during one
    std::mutex deadMutex;
    std::vector<uint64_t> deadBuffers;
    std::vector<KernelEntry*> deadKernels;

    // reused by every dispatch
    std::vector<KernelEntry*> stepKernels;
    std::vector<Use> uses;
    std::vector<std::vector<BufferEntry*>> stepBuffers;

    lua_GpuBackend hooks = {};
    std::function<void(const std::string&)> fallbackHandler;

    mutable std::mutex mutex;

    ~Impl();

    bool init(std::string& error);
    int findMemory(uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred);
    bool createBuffer(VkDeviceSize size, bool deviceLocal, VkBuffer& buffer, VkDeviceMemory& mem, void** mapped);
    void destroyAllocation(Allocation& alloc);
    void destroyBuffer(BufferEntry& entry);

    void destroyPipeline(KernelEntry& entry);
    bool evictLeastRecent();
    void freeDeadBuffers();

    std::string buildPipeline(KernelEntry& entry, const std::string& glsl);
    KernelEntry* kernelFor(lua_State* L, int fn);
    BufferEntry* bufferFor(lua_State* L, int idx, bool create);
    Use* useFor(BufferEntry* entry, void* data);

    int dispatch(lua_State* L, const lua_GpuStep* steps, int nsteps, int outputs, int changed);
    int dispatchLocked(lua_State* L, const lua_GpuStep* steps, int nsteps, int outputs, int changed, std::string& fallback);
    std::string prepare(lua_State* L, const lua_GpuStep* steps, int nsteps, int outputs, int changed);
    void record(const lua_GpuStep* steps, int nsteps, bool batch);
    std::string submit();
    void fallBack(lua_State* L, const lua_GpuStep* steps, int nsteps, int outputs, bool batch, bool ran);
};

#define VK_CHECK(expr) \
    do \
    { \
        VkResult res_ = (expr); \
        if (res_ != VK_SUCCESS) \
        { \
            error = std::string(#expr) + " failed with " + std::to_string(res_); \
            return false; \
        } \
    } while (0)

bool VulkanBackend::Impl::init(std::string& error)
{
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Luau";
    app.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    // prefer discrete GPUs, then integrated ones; software rasterizers are a last resort
    int bestScore = -1;
    VkPhysicalDeviceLimits deviceLimits = {};
    for (VkPhysicalDevice pd : devices)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);

        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> families(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, families.data());

        for (uint32_t q = 0; q < qcount; ++q)
        {
            if (!(families[q].queueFlags & VK_QUEUE_COMPUTE_BIT))
                continue;

            int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                        : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                        : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU            ? 0
                                                                                     : 1;
            if (score > bestScore)
            {
                bestScore = score;
                physical = pd;
                queueFamily = q;
                deviceName = props.deviceName;
                deviceLimits = props.limits;
            }
            break;
        }
    }

    if (!physical)
    {
        error = "no Vulkan device with a compute queue";
        return false;
    }

    if (!options.minimumLimits)
    {
        limits.storageBuffers = deviceLimits.maxPerStageDescriptorStorageBuffers;
        limits.storageRange = deviceLimits.maxStorageBufferRange;
        limits.groupsX = deviceLimits.maxComputeWorkGroupCount[0];
        limits.groupsY = deviceLimits.maxComputeWorkGroupCount[1];
        limits.pushConstants = deviceLimits.maxPushConstantsSize;
    }

    vkGetPhysicalDeviceMemoryProperties(physical, &memory);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(physical, &dci, nullptr, &device));
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = queueFamily;
    VK_CHECK(vkCreateCommandPool(device, &cpci, nullptr, &commandPool));

    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &commandBuffer));

    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));

    // the status word is small and rarely written, so it lives in host memory the GPU writes to directly
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 4;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &status.device));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, status.device, &req);
    int type = findMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    if (type < 0)
    {
        error = "no host visible memory";
        return false;
    }
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &status.deviceMemory));
    VK_CHECK(vkBindBufferMemory(device, status.device, status.deviceMemory, 0));
    VK_CHECK(vkMapMemory(device, status.deviceMemory, 0, 4, 0, &status.mapped));

    hooks.context = this;
    hooks.dispatch = [](void* context, lua_State* L, const lua_GpuStep* steps, int nsteps, int outputs, int changed)
    {
        return static_cast<Impl*>(context)->dispatch(L, steps, nsteps, outputs, changed);
    };
    hooks.status = [](void* context, lua_State* L, int fn)
    {
        Impl* self = static_cast<Impl*>(context);
        std::lock_guard<std::mutex> lock(self->mutex);
        self->freeDeadBuffers();

        KernelEntry* entry = self->kernelFor(L, fn);
        if (!entry->reason.empty())
            lua_pushstring(L, entry->reason.c_str());
        else if (!self->disabledReason.empty())
            lua_pushstring(L, self->disabledReason.c_str());
        else
            lua_pushnil(L);
    };

    return true;
}

#undef VK_CHECK

VulkanBackend::Impl::~Impl()
{
    // a device that may still be executing a dispatch can't be waited for or have its resources destroyed; leak them instead
    if (device && !deviceBusy)
    {
        vkDeviceWaitIdle(device);

        for (auto& [key, entry] : kernels)
            destroyPipeline(*entry);
        for (auto& [id, entry] : buffers)
            destroyBuffer(entry);

        vkDestroyBuffer(device, status.device, nullptr);
        vkFreeMemory(device, status.deviceMemory, nullptr);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
    }

    if (instance && !deviceBusy)
        vkDestroyInstance(instance, nullptr);
}

int VulkanBackend::Impl::findMemory(uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred)
{
    for (int pass = 0; pass < 2; ++pass)
    {
        VkMemoryPropertyFlags flags = pass == 0 ? required | preferred : required;
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        {
            if ((typeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags)
                return int(i);
        }
    }
    return -1;
}

// device-local buffers hold the copies kernels use; host buffers are mapped and cached, since they are compared and read
bool VulkanBackend::Impl::createBuffer(VkDeviceSize size, bool deviceLocal, VkBuffer& buffer, VkDeviceMemory& mem, void** mapped)
{
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | (deviceLocal ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0);
    if (vkCreateBuffer(device, &bci, nullptr, &buffer) != VK_SUCCESS)
    {
        buffer = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buffer, &req);

    int type =
        deviceLocal
            ? findMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)
            : findMemory(
                  req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT
              );
    if (type < 0)
        return false;

    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &mai, nullptr, &mem) != VK_SUCCESS)
    {
        mem = VK_NULL_HANDLE;
        return false;
    }

    if (deviceLocal)
        cachedBytes += size;

    if (vkBindBufferMemory(device, buffer, mem, 0) != VK_SUCCESS)
        return false;

    return !mapped || vkMapMemory(device, mem, 0, size, 0, mapped) == VK_SUCCESS;
}

void VulkanBackend::Impl::destroyAllocation(Allocation& alloc)
{
    if (alloc.deviceMemory)
        cachedBytes -= alloc.deviceSize;
    if (alloc.mapped)
        vkUnmapMemory(device, alloc.stagingMemory);
    vkDestroyBuffer(device, alloc.device, nullptr);
    vkFreeMemory(device, alloc.deviceMemory, nullptr);
    vkDestroyBuffer(device, alloc.staging, nullptr);
    vkFreeMemory(device, alloc.stagingMemory, nullptr);
    alloc = Allocation();
}

void VulkanBackend::Impl::destroyBuffer(BufferEntry& entry)
{
    if (entry.listed)
        lru.erase(entry.lru);
    if (entry.backupMemory)
        cachedBytes -= entry.deviceSize;
    vkDestroyBuffer(device, entry.backup, nullptr);
    vkFreeMemory(device, entry.backupMemory, nullptr);
    destroyAllocation(entry);
    entry = BufferEntry();
}

void VulkanBackend::Impl::destroyPipeline(KernelEntry& entry)
{
    vkDestroyPipeline(device, entry.pipeline, nullptr);
    vkDestroyPipelineLayout(device, entry.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, entry.setLayout, nullptr);
    entry.pipeline = VK_NULL_HANDLE;
    entry.pipelineLayout = VK_NULL_HANDLE;
    entry.setLayout = VK_NULL_HANDLE;
}

// frees the least recently used copy that the current dispatch doesn't use and that holds nothing the Luau buffer lacks;
// buffers are moved to the back of the list when used, so the ones the current dispatch uses are at the back
bool VulkanBackend::Impl::evictLeastRecent()
{
    while (!lru.empty())
    {
        BufferEntry& e = buffers[lru.front()];
        if (e.lastUsed == dispatchCounter)
            return false;

        lru.pop_front();
        e.listed = false;

        // dirty copies come back to the list when they are used again
        if (e.dirty || !e.deviceMemory)
            continue;

        destroyBuffer(e);
        return true;
    }

    return false;
}

void VulkanBackend::Impl::freeDeadBuffers()
{
    std::vector<uint64_t> dead;
    std::vector<KernelEntry*> deadUsers;
    {
        std::lock_guard<std::mutex> lock(deadMutex);
        dead.swap(deadBuffers);
        deadUsers.swap(deadKernels);
    }

    for (KernelEntry* kernel : deadUsers)
    {
        if (--kernel->users > 0)
            continue;

        if (!deviceBusy)
            destroyPipeline(*kernel);
        kernels.erase(kernel->key);
    }

    for (uint64_t id : dead)
    {
        auto it = buffers.find(id);
        if (it == buffers.end())
            continue;
        if (!deviceBusy)
            destroyBuffer(it->second);
        else if (it->second.listed)
            lru.erase(it->second.lru);
        buffers.erase(it);
    }
}

std::string VulkanBackend::Impl::buildPipeline(KernelEntry& entry, const std::string& glsl)
{
    uint32_t bindings = uint32_t(entry.written.size()) + 1;
    if (bindings > limits.storageBuffers)
        return "kernel uses " + std::to_string(bindings) + " storage buffers (its buffers and a status word), the device supports " +
               std::to_string(limits.storageBuffers);
    if (sizeof(PushConstants) > limits.pushConstants)
        return "device doesn't support " + std::to_string(sizeof(PushConstants)) + " bytes of push constants";

    std::vector<uint32_t> spirv;
    std::string error;
    if (!shaders.compileSpirv(glsl, entry.name, spirv, error))
        return "internal error, " + error;

    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv.size() * sizeof(uint32_t);
    smci.pCode = spirv.data();
    VkShaderModule module;
    if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS)
        return "vkCreateShaderModule failed";

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings(bindings);
    for (uint32_t i = 0; i < bindings; ++i)
    {
        layoutBindings[i].binding = i;
        layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[i].descriptorCount = 1;
        layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = bindings;
    dslci.pBindings = layoutBindings.data();

    VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, uint32_t(sizeof(PushConstants))};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &entry.setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &range;

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";

    bool ok = vkCreateDescriptorSetLayout(device, &dslci, nullptr, &entry.setLayout) == VK_SUCCESS &&
              vkCreatePipelineLayout(device, &plci, nullptr, &entry.pipelineLayout) == VK_SUCCESS;
    cpci.layout = entry.pipelineLayout;
    ok = ok && vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &entry.pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(device, module, nullptr);

    if (!ok)
    {
        destroyPipeline(entry);
        return "failed to create the compute pipeline";
    }

    return "";
}

// the kernel of a function, lowered on its first dispatch; functions lowering to the same shader share its pipeline
KernelEntry* VulkanBackend::Impl::kernelFor(lua_State* L, int fn)
{
    int top = lua_gettop(L);
    fn = lua_absindex(L, fn);

    pushWeakTable(L, kKernelsKey);
    lua_pushvalue(L, fn);
    lua_rawget(L, -2);
    KernelHandle* cached = static_cast<KernelHandle*>(lua_touserdata(L, -1));
    lua_settop(L, top);

    if (cached)
        return cached->kernel;

    CodeGen::GpuKernel kernel = lowering(L, fn);

    std::string key = kernel.reason.empty() ? kernel.glsl : "\1" + kernel.name + "\1" + std::to_string(kernel.line) + "\1" + kernel.reason;
    std::unique_ptr<KernelEntry>& entry = kernels[key];

    if (!entry)
    {
        entry = std::make_unique<KernelEntry>();
        entry->key = key;
        entry->name = kernel.name;
        entry->written = kernel.bufferWritten;

        if (!kernel.reason.empty())
            entry->reason = kernel.reason;
        else if (!disabledReason.empty())
            entry->reason = disabledReason;
        else
            entry->reason = buildPipeline(*entry, kernel.glsl);
    }

    KernelEntry* result = entry.get();
    result->users++;

    pushWeakTable(L, kKernelsKey);
    lua_pushvalue(L, fn);
    KernelHandle* handle = static_cast<KernelHandle*>(lua_newuserdatadtor(
        L,
        sizeof(KernelHandle),
        [](lua_State*, void* ud)
        {
            KernelHandle* h = static_cast<KernelHandle*>(ud);
            std::lock_guard<std::mutex> lock(h->impl->deadMutex);
            h->impl->deadKernels.push_back(h->kernel);
        }
    ));
    handle->impl = this;
    handle->kernel = result;
    lua_rawset(L, -3);
    lua_settop(L, top);

    return result;
}

// the device copy of the buffer at stack index idx, allocated when needed if 'create' is set
BufferEntry* VulkanBackend::Impl::bufferFor(lua_State* L, int idx, bool create)
{
    int top = lua_gettop(L);
    idx = lua_absindex(L, idx);

    size_t size = 0;
    lua_tobuffer(L, idx, &size);

    pushWeakTable(L, kBuffersKey);
    lua_pushvalue(L, idx);
    lua_rawget(L, -2);
    BufferHandle* handle = static_cast<BufferHandle*>(lua_touserdata(L, -1));

    if (!handle && create)
    {
        lua_pop(L, 1);
        handle = static_cast<BufferHandle*>(lua_newuserdatadtor(
            L,
            sizeof(BufferHandle),
            [](lua_State*, void* ud)
            {
                BufferHandle* h = static_cast<BufferHandle*>(ud);
                std::lock_guard<std::mutex> lock(h->impl->deadMutex);
                h->impl->deadBuffers.push_back(h->id);
            }
        ));
        handle->impl = this;
        handle->id = ++nextBufferId;

        lua_pushvalue(L, idx);
        lua_pushvalue(L, -2);
        lua_rawset(L, -4);
    }

    lua_settop(L, top);

    if (!handle)
        return nullptr;

    uint64_t id = handle->id;
    BufferEntry& entry = buffers[id];

    if (!entry.deviceMemory && !create)
        return nullptr;

    entry.lastUsed = dispatchCounter;
    if (entry.listed)
        lru.splice(lru.end(), lru, entry.lru);
    else
        entry.lru = lru.insert(lru.end(), id);
    entry.listed = true;

    if (entry.deviceMemory)
        return &entry;

    entry.size = size;
    entry.valid = false;
    entry.dirty = false;

    VkDeviceSize deviceSize = size == 0 ? 4 : (size + 3) & ~size_t(3);

    // copies of collected buffers are freed with them; the budget only bounds copies of live buffers. If allocation still
    // fails, free every copy this dispatch doesn't need and retry
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        while ((attempt > 0 || cachedBytes + deviceSize > options.cacheBytes) && evictLeastRecent())
        {
        }

        entry.deviceSize = deviceSize;
        if (createBuffer(deviceSize, true, entry.device, entry.deviceMemory, nullptr) &&
            createBuffer(deviceSize, false, entry.staging, entry.stagingMemory, &entry.mapped))
            return &entry;

        destroyAllocation(entry);
    }

    return nullptr;
}

Use* VulkanBackend::Impl::useFor(BufferEntry* entry, void* data)
{
    for (Use& u : uses)
    {
        if (u.entry == entry)
            return &u;
    }

    uses.push_back({entry, data, false, false, false});
    return &uses.back();
}

int VulkanBackend::Impl::dispatch(lua_State* L, const lua_GpuStep* steps, int nsteps, int outputs, int changed)
{
    std::string fallback;
    std::function<void(const std::string&)> handler;
    int result;

    {
        std::lock_guard<std::mutex> lock(mutex);
        result = dispatchLocked(L, steps, nsteps, outputs, changed, fallback);
        handler = fallbackHandler;
    }

    // outside of the lock, so the handler can use the backend; it only reports, so its failures don't affect the dispatch
    if (!fallback.empty() && handler)
    {
        try
        {
            handler(fallback);
        }
        catch (...)
        {
        }
    }

    return result;
}

int VulkanBackend::Impl::dispatchLocked(lua_State* L, const lua_GpuStep* steps, int nsteps, int outputs, int changed, std::string& fallback)
{
    stats.dispatches++;
    freeDeadBuffers();

    bool batch = outputs != 0;
    dispatchCounter++;

    stepKernels.clear();
    std::string reason = disabledReason;

    for (int s = 0; s < nsteps && reason.empty(); ++s)
    {
        if (!steps[s].safe)
        {
            reason = "the function's environment isn't safe";
            break;
        }

        KernelEntry* kernel = kernelFor(L, steps[s].fn);
        stepKernels.push_back(kernel);

        if (!kernel->reason.empty())
        {
            // reported once per kernel, and on dispatch rather than when lowering since @gpu helpers aren't dispatched
            if (!kernel->warned)
                fallback = "@gpu function '" + kernel->name + "' runs in the VM: " + kernel->reason;
            kernel->warned = true;
            reason = kernel->reason;
        }
        else if (size_t(steps[s].nargs) < kernel->written.size())
        {
            reason = "missing buffers"; // the VM raises the error the script expects
        }
    }

    if (reason.empty())
    {
        reason = prepare(L, steps, nsteps, outputs, changed);

        // limits and memory are reported once per kernel, like lowering failures
        if (!reason.empty() && nsteps > 0 && !stepKernels[0]->warned)
        {
            fallback = "@gpu function '" + stepKernels[0]->name + "' runs in the VM: " + reason;
            stepKernels[0]->warned = true;
        }
    }

    if (!reason.empty())
    {
        fallBack(L, steps, nsteps, outputs, batch, false);
        return 0;
    }

    record(steps, nsteps, batch);
    reason = submit();

    if (!reason.empty())
    {
        // the device is disabled; what it holds is lost, the VM runs with the Luau contents
        for (Use& u : uses)
            u.entry->valid = u.entry->dirty = false;
        return 0;
    }

    if (*(volatile uint32_t*)status.mapped != 0)
    {
        // an error on the GPU (out of bounds or unaligned access, integer overflow, loop budget...) discards the results; the
        // VM then reruns the dispatch and raises the error Luau would
        stats.fallbacks++;
        fallBack(L, steps, nsteps, outputs, batch, true);
        return 0;
    }

    for (Use& u : uses)
    {
        if (u.output && (u.written || u.entry->dirty))
        {
            memcpy(u.data, u.entry->mapped, u.entry->size);
            stats.downloadedBytes += u.entry->size;
            u.entry->dirty = false;
        }
        else if (u.written)
        {
            u.entry->dirty = true;
        }
    }

    stats.gpuDispatches++;
    return 1;
}

// checks the device limits and gets the device copy of every buffer the dispatch uses or must bring up to date
std::string VulkanBackend::Impl::prepare(lua_State* L, const lua_GpuStep* steps, int nsteps, int outputs, int changed)
{
    uses.clear();
    stepBuffers.resize(nsteps);

    for (int s = 0; s < nsteps; ++s)
    {
        uint64_t groups = (uint64_t(steps[s].count) + kWorkgroupSize - 1) / kWorkgroupSize;
        uint64_t groupsX = groups < limits.groupsX ? groups : limits.groupsX;
        if (groupsX > 0 && (groups + groupsX - 1) / groupsX > limits.groupsY)
            return std::to_string(steps[s].count) + " invocations exceed the device's workgroup limits";

        const KernelEntry* kernel = stepKernels[s];
        stepBuffers[s].clear();

        for (size_t i = 0; i < kernel->written.size(); ++i)
        {
            size_t len = 0;
            void* data = lua_tobuffer(L, steps[s].args + int(i), &len);

            if (((uint64_t(len) + 3) & ~uint64_t(3)) > limits.storageRange)
                return "a buffer of " + std::to_string(len) + " bytes exceeds the device's storage buffer range of " +
                       std::to_string(limits.storageRange) + " bytes";

            BufferEntry* entry = bufferFor(L, steps[s].args + int(i), true);
            if (!entry)
                return "not enough device memory for a buffer of " + std::to_string(len) + " bytes";

            stepBuffers[s].push_back(entry);
            Use* u = useFor(entry, data);
            u->written |= kernel->written[i];
            // a single dispatch reads every buffer from Luau and leaves every written buffer up to date in Luau
            u->output |= !outputs && kernel->written[i];
        }
    }

    for (int t : {outputs, changed})
    {
        if (!t || !lua_istable(L, t))
            continue;

        for (int i = 1, n = lua_objlen(L, t); i <= n; ++i)
        {
            lua_rawgeti(L, t, i);
            void* data = lua_tobuffer(L, -1, nullptr);
            // a buffer without a device copy already holds its contents
            BufferEntry* entry = bufferFor(L, -1, false);
            lua_pop(L, 1);

            if (!entry)
                continue;

            if (t == outputs)
            {
                useFor(entry, data)->output = true;
            }
            else
            {
                bool used = false;
                for (Use& u : uses)
                {
                    if (u.entry == entry)
                        u.changed = used = true;
                }

                // modified by Luau code: the next dispatch that uses it uploads it
                if (!used)
                    entry->valid = entry->dirty = false;
            }
        }
    }

    // copies that are newer than their Luau buffer and that this dispatch writes are backed up, to restore them on failure
    for (Use& u : uses)
    {
        BufferEntry* e = u.entry;
        if (outputs && e->dirty && u.written && !u.changed && !e->backup && !createBuffer(e->deviceSize, true, e->backup, e->backupMemory, nullptr))
            return "not enough device memory to back up a buffer of " + std::to_string(e->size) + " bytes";
    }

    if (descriptorSets < uint32_t(nsteps))
    {
        if (descriptorPool)
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        uint32_t sets = descriptorSets ? descriptorSets : 16;
        while (sets < uint32_t(nsteps))
            sets *= 2;

        VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sets * (kMaxKernelBuffers + 1)};
        VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = sets;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device, &dpci, nullptr, &descriptorPool) != VK_SUCCESS)
        {
            descriptorPool = VK_NULL_HANDLE;
            descriptorSets = 0;
            return "failed to create a descriptor pool";
        }
        descriptorSets = sets;
    }

    return "";
}

void VulkanBackend::Impl::record(const lua_GpuStep* steps, int nsteps, bool batch)
{
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &begin);

    for (Use& u : uses)
    {
        BufferEntry* e = u.entry;

        // a single dispatch compares every buffer with what the device holds, a batch trusts buffers that aren't listed as
        // changed; in both, the Luau contents of a buffer win over a copy the Luau buffer is behind
        bool upload = !e->valid || (batch ? u.changed : e->dirty || memcmp(e->mapped, u.data, e->size) != 0);

        if (upload)
        {
            memcpy(e->mapped, u.data, e->size);
            copyBuffer(commandBuffer, e->staging, e->device, e->deviceSize);
            e->valid = true;
            e->dirty = false;
            stats.uploadedBytes += e->size;
        }
        else if (e->dirty && u.written)
        {
            copyBuffer(commandBuffer, e->device, e->backup, e->deviceSize);
        }
    }

    barrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    );

    // the pool only exists once a batch had steps; an empty batch (a download) can come first
    if (descriptorPool)
        vkResetDescriptorPool(device, descriptorPool, 0);

    for (int s = 0; s < nsteps; ++s)
    {
        const KernelEntry* kernel = stepKernels[s];
        const std::vector<BufferEntry*>& bound = stepBuffers[s];

        if (steps[s].count == 0)
            continue;

        VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = descriptorPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &kernel->setLayout;
        VkDescriptorSet set;
        vkAllocateDescriptorSets(device, &dsai, &set);

        // bindings follow the argument order, so a buffer passed twice is bound twice; the status word comes last
        VkDescriptorBufferInfo infos[kMaxKernelBuffers + 1] = {};
        VkWriteDescriptorSet writes[kMaxKernelBuffers + 1] = {};
        size_t bindings = bound.size() + 1;
        for (size_t i = 0; i < bindings; ++i)
        {
            infos[i].buffer = i < bound.size() ? bound[i]->device : status.device;
            infos[i].range = VK_WHOLE_SIZE;

            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = set;
            writes[i].dstBinding = uint32_t(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, uint32_t(bindings), writes, 0, nullptr);

        PushConstants push = {};
        push.count = uint32_t(steps[s].count);
        push.loopBudget = options.loopBudget;
        for (size_t i = 0; i < bound.size(); ++i)
            push.lengths[i] = uint32_t(bound[i]->size);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(commandBuffer, kernel->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        uint32_t groups = uint32_t((uint64_t(push.count) + kWorkgroupSize - 1) / kWorkgroupSize);
        uint32_t groupsX = groups < limits.groupsX ? groups : limits.groupsX;
        vkCmdDispatch(commandBuffer, groupsX, (groups + groupsX - 1) / groupsX, 1);

        // each step sees what the previous ones wrote
        barrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT
        );
    }

    for (Use& u : uses)
    {
        if (u.output && (u.written || u.entry->dirty))
            copyBuffer(commandBuffer, u.entry->device, u.entry->staging, u.entry->deviceSize);
    }

    barrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_HOST_READ_BIT
    );

    vkEndCommandBuffer(commandBuffer);
}

// runs the recorded commands; a timeout or a device error disables the backend
std::string VulkanBackend::Impl::submit()
{
    *(volatile uint32_t*)status.mapped = 0;

    VkSubmitInfo info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.commandBufferCount = 1;
    info.pCommandBuffers = &commandBuffer;
    vkResetFences(device, 1, &fence);

    VkResult res = vkQueueSubmit(queue, 1, &info, fence);
    if (res == VK_SUCCESS)
        res = vkWaitForFences(device, 1, &fence, VK_TRUE, uint64_t(options.timeoutMs) * 1000000);

    if (res == VK_SUCCESS)
        return "";

    deviceBusy = res == VK_TIMEOUT;
    disabledReason = res == VK_TIMEOUT ? "the GPU backend was disabled after a dispatch exceeded " + std::to_string(options.timeoutMs) + " ms"
                                       : "the GPU backend was disabled after a device error (" + std::to_string(res) + ")";
    return disabledReason;
}

// before the VM runs a dispatch, Luau buffers must hold the contents the dispatch reads, and afterwards the device copies
// can't be trusted; 'ran' is set when the dispatch ran on the GPU and changed the copies it writes
void VulkanBackend::Impl::fallBack(lua_State* L, const lua_GpuStep* steps, int nsteps, int outputs, bool batch, bool ran)
{
    if (!ran)
    {
        uses.clear();
        for (int s = 0; s < nsteps; ++s)
        {
            for (int i = 0; i < steps[s].nargs; ++i)
            {
                if (BufferEntry* entry = bufferFor(L, steps[s].args + i, false))
                    useFor(entry, lua_tobuffer(L, steps[s].args + i, nullptr));
            }
        }

        if (outputs && lua_istable(L, outputs))
        {
            for (int i = 1, n = lua_objlen(L, outputs); i <= n; ++i)
            {
                lua_rawgeti(L, outputs, i);
                if (BufferEntry* entry = bufferFor(L, -1, false))
                    useFor(entry, lua_tobuffer(L, -1, nullptr));
                lua_pop(L, 1);
            }
        }
    }

    // a single dispatch reads Luau contents anyway; a batch reads what earlier batches left on the device
    bool restore = false;
    if (batch && disabledReason.empty())
    {
        VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(commandBuffer, 0);
        vkBeginCommandBuffer(commandBuffer, &begin);

        for (Use& u : uses)
        {
            if (!u.entry->dirty)
                continue;
            copyBuffer(commandBuffer, ran && u.written ? u.entry->backup : u.entry->device, u.entry->staging, u.entry->deviceSize);
            restore = true;
        }

        barrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        vkEndCommandBuffer(commandBuffer);

        if (restore && submit().empty())
        {
            for (Use& u : uses)
            {
                if (!u.entry->dirty)
                    continue;
                memcpy(u.data, u.entry->mapped, u.entry->size);
                stats.downloadedBytes += u.entry->size;
            }
        }
    }

    // the VM may write any of these buffers
    for (Use& u : uses)
        u.entry->valid = u.entry->dirty = false;
}

std::unique_ptr<VulkanBackend> VulkanBackend::create(std::string& error, const VulkanOptions& options)
{
    std::unique_ptr<VulkanBackend> backend(new VulkanBackend());
    backend->impl = std::make_unique<Impl>();
    backend->impl->options = options;
    if (!backend->impl->init(error))
        return nullptr;
    return backend;
}

VulkanBackend::~VulkanBackend() = default;

void VulkanBackend::setFallbackHandler(std::function<void(const std::string& message)> handler)
{
    std::lock_guard<std::mutex> lock(impl->mutex);
    impl->fallbackHandler = std::move(handler);
}

void VulkanBackend::setLowering(std::function<CodeGen::GpuKernel(lua_State* L, int idx)> lowering)
{
    std::lock_guard<std::mutex> lock(impl->mutex);
    impl->lowering = std::move(lowering);
}

void VulkanBackend::install(lua_State* L)
{
    lua_setgpubackend(L, &impl->hooks);
}

const std::string& VulkanBackend::deviceName() const
{
    return impl->deviceName;
}

DispatchStats VulkanBackend::stats() const
{
    std::lock_guard<std::mutex> lock(impl->mutex);
    DispatchStats s = impl->stats;
    s.cachedBytes = impl->cachedBytes;
    s.kernels = impl->kernels.size();
    return s;
}

} // namespace Gpu
} // namespace Luau

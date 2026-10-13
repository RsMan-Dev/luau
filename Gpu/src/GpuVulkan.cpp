// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/GpuVulkan.h"

#include "Luau/GpuShaderCompiler.h"

#include "lua.h"
#include "lualib.h"

#include <vulkan/vulkan.h>

#include <string.h>

#include <map>
#include <mutex>
#include <unordered_map>

namespace Luau
{
namespace Gpu
{

namespace
{

constexpr uint64_t kEvictAfterDispatches = 2000;

struct KernelEntry
{
    std::string name;
    std::string reason;
    bool warned = false;
    std::vector<bool> written;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

// device copy of a Luau buffer; the staging memory mirrors what the device holds while 'valid' is set
struct BufferEntry
{
    size_t size = 0;
    VkDeviceSize deviceSize = 0;

    VkBuffer device = VK_NULL_HANDLE;
    VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    void* mapped = nullptr;

    bool valid = false;
    uint64_t lastUsed = 0;
};

// a Luau buffer bound to a dispatch; a buffer passed several times is transferred once
struct Transfer
{
    BufferEntry* entry;
    void* data;
    bool written;
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

    BufferEntry status;

    ShaderCompiler shaders;

    std::string deviceName;
    DispatchStats stats;
    uint64_t dispatchCounter = 0;

    // set when the device stopped responding or was lost; every dispatch then runs in the VM
    std::string disabledReason;
    bool deviceBusy = false;

    std::map<std::pair<std::string, int>, KernelEntry> kernels;
    std::unordered_map<const void*, BufferEntry> buffers;
    uint64_t cachedBytes = 0;

    lua_GpuBackend hooks = {};
    std::function<void(const std::string&)> fallbackHandler;

    mutable std::mutex mutex;

    ~Impl();

    bool init(std::string& error);
    int findMemory(uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred);
    bool createBuffer(BufferEntry& entry, bool deviceLocal);
    void destroyBuffer(BufferEntry& entry);
    void destroyPipeline(KernelEntry& entry);
    BufferEntry* bufferFor(const void* data, size_t size);
    bool evictLeastRecent();
    void evict();

    std::string buildPipeline(KernelEntry& entry, const Kernel& kernel);
    KernelEntry* lookup(lua_State* L, int fnidx);

    int dispatch(lua_State* L, int fnidx, int n, int argidx, int nargs);
    int dispatchLocked(lua_State* L, int fnidx, int n, int argidx, int nargs, std::string& fallback);
    std::string prepare(
        lua_State* L,
        const KernelEntry& kernel,
        int argidx,
        std::vector<BufferEntry*>& bound,
        std::vector<Transfer>& transfers,
        PushConstants& push
    );
    void record(const KernelEntry& kernel, const std::vector<BufferEntry*>& bound, const std::vector<Transfer>& transfers, const PushConstants& push);
    std::string execute(const std::vector<Transfer>& transfers, bool& discarded);
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

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxKernelBuffers + 1};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &descriptorPool));

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
    hooks.dispatch = [](void* context, lua_State* L, int fnidx, int n, int argidx, int nargs)
    {
        return static_cast<Impl*>(context)->dispatch(L, fnidx, n, argidx, nargs);
    };
    hooks.status = [](void* context, lua_State* L, int fnidx)
    {
        Impl* self = static_cast<Impl*>(context);
        std::lock_guard<std::mutex> lock(self->mutex);

        KernelEntry* entry = self->lookup(L, fnidx);
        if (!entry)
            lua_pushstring(L, "function is not marked @gpu, or its chunk wasn't registered with the GPU backend from the same source");
        else if (!entry->reason.empty())
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
            destroyPipeline(entry);
        for (auto& [key, entry] : buffers)
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

bool VulkanBackend::Impl::createBuffer(BufferEntry& entry, bool deviceLocal)
{
    VkBuffer& buffer = deviceLocal ? entry.device : entry.staging;
    VkDeviceMemory& mem = deviceLocal ? entry.deviceMemory : entry.stagingMemory;

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = entry.deviceSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | (deviceLocal ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0);
    if (vkCreateBuffer(device, &bci, nullptr, &buffer) != VK_SUCCESS)
    {
        buffer = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buffer, &req);

    // staging memory is compared against the Luau buffer on every dispatch, so it must be cached for reads
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
        cachedBytes += entry.deviceSize;

    if (vkBindBufferMemory(device, buffer, mem, 0) != VK_SUCCESS)
        return false;

    return deviceLocal || vkMapMemory(device, mem, 0, entry.deviceSize, 0, &entry.mapped) == VK_SUCCESS;
}

void VulkanBackend::Impl::destroyBuffer(BufferEntry& entry)
{
    if (entry.deviceMemory)
        cachedBytes -= entry.deviceSize;
    if (entry.mapped)
        vkUnmapMemory(device, entry.stagingMemory);
    vkDestroyBuffer(device, entry.device, nullptr);
    vkFreeMemory(device, entry.deviceMemory, nullptr);
    vkDestroyBuffer(device, entry.staging, nullptr);
    vkFreeMemory(device, entry.stagingMemory, nullptr);
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

// Luau buffers are identified by their data pointer; a recycled address is harmless since contents are always compared
BufferEntry* VulkanBackend::Impl::bufferFor(const void* data, size_t size)
{
    auto it = buffers.find(data);
    if (it != buffers.end() && it->second.size != size)
    {
        destroyBuffer(it->second);
        buffers.erase(it);
        it = buffers.end();
    }

    if (it == buffers.end())
    {
        BufferEntry entry;
        entry.size = size;
        entry.deviceSize = size == 0 ? 4 : (size + 3) & ~size_t(3);

        // stay within the cache budget; if allocation still fails, free every copy this dispatch doesn't use and retry
        while (cachedBytes + entry.deviceSize > options.cacheBytes && evictLeastRecent())
        {
        }

        bool created = createBuffer(entry, true) && createBuffer(entry, false);
        if (!created)
        {
            destroyBuffer(entry);
            while (evictLeastRecent())
            {
            }
            created = createBuffer(entry, true) && createBuffer(entry, false);
        }

        if (!created)
        {
            destroyBuffer(entry);
            return nullptr;
        }

        it = buffers.emplace(data, entry).first;
    }

    it->second.lastUsed = dispatchCounter;
    return &it->second;
}

// frees the least recently used copy that the current dispatch doesn't use
bool VulkanBackend::Impl::evictLeastRecent()
{
    auto victim = buffers.end();
    for (auto it = buffers.begin(); it != buffers.end(); ++it)
    {
        if (it->second.lastUsed != dispatchCounter && (victim == buffers.end() || it->second.lastUsed < victim->second.lastUsed))
            victim = it;
    }

    if (victim == buffers.end())
        return false;

    destroyBuffer(victim->second);
    buffers.erase(victim);
    return true;
}

void VulkanBackend::Impl::evict()
{
    for (auto it = buffers.begin(); it != buffers.end();)
    {
        if (dispatchCounter - it->second.lastUsed > kEvictAfterDispatches)
        {
            destroyBuffer(it->second);
            it = buffers.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

std::string VulkanBackend::Impl::buildPipeline(KernelEntry& entry, const Kernel& kernel)
{
    uint32_t bindings = uint32_t(kernel.bufferWritten.size()) + 1;
    if (bindings > limits.storageBuffers)
        return "kernel uses " + std::to_string(bindings) + " storage buffers (its buffers and a status word), the device supports " +
               std::to_string(limits.storageBuffers);
    if (sizeof(PushConstants) > limits.pushConstants)
        return "device doesn't support " + std::to_string(sizeof(PushConstants)) + " bytes of push constants";

    std::vector<uint32_t> spirv;
    std::string error;
    if (!shaders.compileSpirv(kernel, spirv, error))
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

KernelEntry* VulkanBackend::Impl::lookup(lua_State* L, int fnidx)
{
    lua_pushvalue(L, fnidx);
    lua_Debug ar = {};
    int ok = lua_getinfo(L, -1, "sa", &ar);
    lua_pop(L, 1);

    if (!ok || !ar.source || ar.linedefined <= 0)
        return nullptr;

    auto it = kernels.find({ar.source, ar.linedefined});
    if (it == kernels.end())
        return nullptr;

    // kernels are found by line, so a function compiled from another source than the one registered must not run the shader;
    // kernels that run in the VM anyway keep reporting their own reason
    if (it->second.reason.empty() && (ar.isvararg || size_t(ar.nparams) != it->second.written.size() + 1))
        return nullptr;

    return &it->second;
}

int VulkanBackend::Impl::dispatch(lua_State* L, int fnidx, int n, int argidx, int nargs)
{
    std::string fallback;
    std::function<void(const std::string&)> handler;
    int result;

    {
        std::lock_guard<std::mutex> lock(mutex);
        result = dispatchLocked(L, fnidx, n, argidx, nargs, fallback);
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

int VulkanBackend::Impl::dispatchLocked(lua_State* L, int fnidx, int n, int argidx, int nargs, std::string& fallback)
{
    stats.dispatches++;

    KernelEntry* kernel = lookup(L, fnidx);
    if (!kernel)
        return 0;

    std::string reason = !kernel->reason.empty() ? kernel->reason : disabledReason;

    if (reason.empty())
    {
        if (size_t(nargs) < kernel->written.size())
            return 0; // missing buffers; the VM raises the error the script expects
        if (n == 0)
            return 1;
    }

    std::vector<BufferEntry*> bound;
    std::vector<Transfer> transfers;
    PushConstants push = {};
    push.count = uint32_t(n);
    push.loopBudget = options.loopBudget;

    if (reason.empty())
        reason = prepare(L, *kernel, argidx, bound, transfers, push);

    bool discarded = false;
    if (reason.empty())
    {
        record(*kernel, bound, transfers, push);
        reason = execute(transfers, discarded);
    }

    if (!reason.empty())
    {
        // reported once per kernel, and on dispatch rather than at load since @gpu helpers aren't meant to be dispatched
        if (!kernel->warned)
            fallback = "@gpu function '" + kernel->name + "' runs in the VM: " + reason;
        kernel->warned = true;
        return 0;
    }

    if (discarded)
        return 0;

    stats.gpuDispatches++;
    if (dispatchCounter % 256 == 0)
        evict();

    return 1;
}

// checks the device limits for this dispatch and gets the device copy of every buffer
std::string VulkanBackend::Impl::prepare(
    lua_State* L,
    const KernelEntry& kernel,
    int argidx,
    std::vector<BufferEntry*>& bound,
    std::vector<Transfer>& transfers,
    PushConstants& push
)
{
    uint64_t groups = (uint64_t(push.count) + kWorkgroupSize - 1) / kWorkgroupSize;
    uint64_t groupsX = groups < limits.groupsX ? groups : limits.groupsX;
    if ((groups + groupsX - 1) / groupsX > limits.groupsY)
        return std::to_string(push.count) + " invocations exceed the device's workgroup limits";

    dispatchCounter++;

    for (size_t i = 0; i < kernel.written.size(); ++i)
    {
        size_t len = 0;
        void* data = lua_tobuffer(L, argidx + int(i), &len);

        if (((uint64_t(len) + 3) & ~uint64_t(3)) > limits.storageRange)
            return "a buffer of " + std::to_string(len) + " bytes exceeds the device's storage buffer range of " +
                   std::to_string(limits.storageRange) + " bytes";

        BufferEntry* entry = bufferFor(data, len);
        if (!entry)
            return "not enough device memory for a buffer of " + std::to_string(len) + " bytes";

        push.lengths[i] = uint32_t(len);
        bound.push_back(entry);

        bool seen = false;
        for (Transfer& t : transfers)
        {
            if (t.entry == entry)
            {
                t.written |= kernel.written[i];
                seen = true;
            }
        }
        if (!seen)
            transfers.push_back({entry, data, kernel.written[i]});
    }

    return "";
}

void VulkanBackend::Impl::record(
    const KernelEntry& kernel,
    const std::vector<BufferEntry*>& bound,
    const std::vector<Transfer>& transfers,
    const PushConstants& push
)
{
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &begin);

    for (const Transfer& t : transfers)
    {
        BufferEntry* entry = t.entry;
        if (entry->valid && memcmp(entry->mapped, t.data, entry->size) == 0)
            continue;

        memcpy(entry->mapped, t.data, entry->size);
        VkBufferCopy copy = {0, 0, entry->deviceSize};
        vkCmdCopyBuffer(commandBuffer, entry->staging, entry->device, 1, &copy);
        entry->valid = true;
        stats.uploadedBytes += entry->size;
    }

    barrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    );

    vkResetDescriptorPool(device, descriptorPool, 0);
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descriptorPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &kernel.setLayout;
    VkDescriptorSet set;
    vkAllocateDescriptorSets(device, &dsai, &set);

    // bindings follow the argument order, so a buffer passed twice is bound twice; the status word comes last
    size_t bindings = bound.size() + 1;
    std::vector<VkDescriptorBufferInfo> infos(bindings);
    std::vector<VkWriteDescriptorSet> writes(bindings);
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
    vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(commandBuffer, kernel.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t groups = uint32_t((uint64_t(push.count) + kWorkgroupSize - 1) / kWorkgroupSize);
    uint32_t groupsX = groups < limits.groupsX ? groups : limits.groupsX;
    vkCmdDispatch(commandBuffer, groupsX, (groups + groupsX - 1) / groupsX, 1);

    barrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
    );

    for (const Transfer& t : transfers)
    {
        if (!t.written)
            continue;
        VkBufferCopy copy = {0, 0, t.entry->deviceSize};
        vkCmdCopyBuffer(commandBuffer, t.entry->device, t.entry->staging, 1, &copy);
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

// runs the recorded dispatch and copies the results back unless the GPU reported an error, in which case 'discarded' is set
// and the VM reruns the dispatch to raise the error Luau would
std::string VulkanBackend::Impl::execute(const std::vector<Transfer>& transfers, bool& discarded)
{
    auto invalidateWritten = [&]()
    {
        for (const Transfer& t : transfers)
        {
            if (t.written)
                t.entry->valid = false;
        }
    };

    *(volatile uint32_t*)status.mapped = 0;

    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer;
    vkResetFences(device, 1, &fence);

    VkResult res = vkQueueSubmit(queue, 1, &submit, fence);
    if (res == VK_SUCCESS)
        res = vkWaitForFences(device, 1, &fence, VK_TRUE, uint64_t(options.timeoutMs) * 1000000);

    if (res != VK_SUCCESS)
    {
        invalidateWritten();
        deviceBusy = res == VK_TIMEOUT;
        disabledReason = res == VK_TIMEOUT ? "the GPU backend was disabled after a dispatch exceeded " + std::to_string(options.timeoutMs) + " ms"
                                           : "the GPU backend was disabled after a device error (" + std::to_string(res) + ")";
        return disabledReason;
    }

    if (*(volatile uint32_t*)status.mapped != 0)
    {
        invalidateWritten();
        stats.fallbacks++;
        discarded = true;
        return "";
    }

    for (const Transfer& t : transfers)
    {
        if (!t.written)
            continue;
        memcpy(t.data, t.entry->mapped, t.entry->size);
        stats.downloadedBytes += t.entry->size;
    }

    return "";
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

std::vector<Kernel> VulkanBackend::load(const std::string& chunkname, const char* source, size_t size)
{
    std::vector<Kernel> kernels = translate(source, size);
    loadKernels(chunkname, kernels);
    return kernels;
}

void VulkanBackend::loadKernels(const std::string& chunkname, std::vector<Kernel>& kernels)
{
    std::lock_guard<std::mutex> lock(impl->mutex);

    // reloading a chunk replaces all of its kernels, so that a function now at the line of a former kernel doesn't run it;
    // pipelines a hung device may still use are leaked rather than destroyed
    for (auto it = impl->kernels.begin(); it != impl->kernels.end();)
    {
        if (it->first.first == chunkname)
        {
            if (!impl->deviceBusy)
                impl->destroyPipeline(it->second);
            it = impl->kernels.erase(it);
        }
        else
        {
            ++it;
        }
    }

    for (Kernel& kernel : kernels)
    {
        KernelEntry& entry = impl->kernels[{chunkname, kernel.line}];
        entry.name = kernel.name;
        entry.written = kernel.bufferWritten;

        if (!kernel.reason.empty())
            entry.reason = "line " + std::to_string(kernel.reasonLocation.begin.line + 1) + ": " + kernel.reason;
        else if (!impl->disabledReason.empty())
            entry.reason = impl->disabledReason;
        else
            entry.reason = impl->buildPipeline(entry, kernel);

        kernel.reason = entry.reason;
    }
}

void VulkanBackend::setFallbackHandler(std::function<void(const std::string& message)> handler)
{
    std::lock_guard<std::mutex> lock(impl->mutex);
    impl->fallbackHandler = std::move(handler);
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
    return s;
}

} // namespace Gpu
} // namespace Luau

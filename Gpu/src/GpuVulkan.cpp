// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/GpuVulkan.h"

#include "Luau/GpuShaderCompiler.h"

#include "lua.h"
#include "lualib.h"

#include <vulkan/vulkan.h>

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Luau
{
namespace Gpu
{

namespace
{

// runs tasks on a few host threads: comparing and copying buffers is limited by memory bandwidth, which one thread doesn't use
// up
class HostThreads
{
public:
    HostThreads()
    {
        unsigned n = std::thread::hardware_concurrency();
        n = n < 2 ? 0 : n > 8 ? 7 : n - 1;
        for (unsigned i = 0; i < n; ++i)
            workers.emplace_back(
                [this]
                {
                    work();
                }
            );
    }

    ~HostThreads()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        for (std::thread& t : workers)
            t.join();
    }

    // calls task(i) for i = 0 .. count-1, on this thread and the workers, and returns once every call returned; a few tasks
    // run on this thread only, since waking the workers costs more than they take
    void run(size_t count, const std::function<void(size_t)>& task)
    {
        if (workers.empty() || count < 32)
        {
            for (size_t i = 0; i < count; ++i)
                task(i);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            current = &task;
            total = count;
            next = 0;
            pending = count;
            generation++;
        }
        wake.notify_all();
        take();

        std::unique_lock<std::mutex> lock(mutex);
        finished.wait(
            lock,
            [this]
            {
                return pending == 0;
            }
        );
        current = nullptr;
    }

private:
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable wake, finished;
    const std::function<void(size_t)>* current = nullptr;
    size_t total = 0, next = 0, pending = 0;
    uint64_t generation = 0;
    bool stopping = false;

    void take()
    {
        for (;;)
        {
            size_t i;
            const std::function<void(size_t)>* task;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!current || next >= total)
                    return;
                i = next++;
                task = current;
            }

            (*task)(i);

            std::lock_guard<std::mutex> lock(mutex);
            if (--pending == 0)
                finished.notify_one();
        }
    }

    void work()
    {
        uint64_t seen = 0;
        for (;;)
        {
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(
                    lock,
                    [&]
                    {
                        return stopping || generation != seen;
                    }
                );
                if (stopping)
                    return;
                seen = generation;
            }
            take();
        }
    }
};

// registry tables, weak on their keys, that map functions to kernels and buffers to the device copies of them
const char* const kKernelsKey = "_GPUKERNELS";
const char* const kBuffersKey = "_GPUBUFFERS";

// kernels larger than this are refused, see kernelFor
constexpr size_t kMaxKernelGlsl = size_t(2) << 20;

struct KernelEntry
{
    std::string key;
    // functions using the kernel, in every state; it's freed with the last one
    int users = 0;

    std::string name;
    std::string reason;
    std::vector<bool> written;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    // the pipeline is created on a compile thread, which sets kReady once it's done and compileError when it failed; a
    // kernel whose functions were collected meanwhile is kAbandoned, and whichever of the two comes last destroys it
    static constexpr uint32_t kReady = 1, kAbandoned = 2;
    std::atomic<uint32_t> flags{0};
    std::string compileError;
    std::string glsl; // until the compile thread takes it

    // the kernel's exact variant, compiled the first time a run of it needs it, and how many of the next runs use it
    // directly: data that needed it once, as denormal numbers, likely needs it again
    std::shared_ptr<KernelEntry> exact;
    std::string exactGlsl;
    int preferExact = 0;

    bool ready() const
    {
        return (flags.load(std::memory_order_acquire) & kReady) != 0;
    }
};

void destroyPipeline(VkDevice device, KernelEntry& entry)
{
    vkDestroyPipeline(device, entry.pipeline, nullptr);
    vkDestroyPipelineLayout(device, entry.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, entry.setLayout, nullptr);
    entry.pipeline = VK_NULL_HANDLE;
    entry.pipelineLayout = VK_NULL_HANDLE;
    entry.setLayout = VK_NULL_HANDLE;
}

std::string createPipeline(
    VkDevice device,
    VkPipelineCache cache,
    uint32_t pushConstants,
    ShaderCompiler& shaders,
    KernelEntry& entry,
    const std::string& glsl
)
{
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

    uint32_t bindings = uint32_t(entry.written.size()) + 2;
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

    VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstants};
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
    ok = ok && vkCreateComputePipelines(device, cache, 1, &cpci, nullptr, &entry.pipeline) == VK_SUCCESS;
    vkDestroyShaderModule(device, module, nullptr);

    if (!ok)
    {
        destroyPipeline(device, entry);
        return "failed to create the compute pipeline";
    }

    return "";
}

// creates pipelines on a few host threads: drivers take a time growing faster than the size of the kernel, minutes on large
// ones, which runs can't wait for. What the threads use is shared with them, so that a backend destroyed while a driver
// compiles doesn't take it away: the thread is left to finish, then destroys what it made
class PipelineCompiler
{
public:
    PipelineCompiler(VkDevice device, VkPipelineCache cache, uint32_t pushConstants)
        : shared(std::make_shared<Shared>())
    {
        shared->device = device;
        shared->cache = cache;
        shared->pushConstants = pushConstants;

        unsigned n = std::thread::hardware_concurrency();
        n = n < 4 ? 1 : n < 8 ? 2 : 3;
        for (unsigned i = 0; i < n; ++i)
            workers.emplace_back(work, shared);
    }

    // returns whether a thread was still creating a pipeline, which the device must then outlive
    bool shutdown()
    {
        bool busy;
        {
            std::lock_guard<std::mutex> lock(shared->mutex);
            shared->stopping = true;
            shared->queue.clear();
            busy = shared->busy > 0;
        }
        shared->wake.notify_all();

        for (std::thread& t : workers)
        {
            if (busy)
                t.detach();
            else
                t.join();
        }
        workers.clear();
        return busy;
    }

    void submit(std::shared_ptr<KernelEntry> entry)
    {
        {
            std::lock_guard<std::mutex> lock(shared->mutex);
            shared->queue.push_back(std::move(entry));
        }
        shared->wake.notify_one();
    }

    void wait(const KernelEntry& entry)
    {
        std::unique_lock<std::mutex> lock(shared->mutex);
        shared->done.wait(
            lock,
            [&]
            {
                return entry.ready();
            }
        );
    }

private:
    struct Shared
    {
        VkDevice device = VK_NULL_HANDLE;
        VkPipelineCache cache = VK_NULL_HANDLE;
        uint32_t pushConstants = 0;

        std::mutex mutex;
        std::condition_variable wake, done;
        std::list<std::shared_ptr<KernelEntry>> queue;
        int busy = 0;
        bool stopping = false;
    };

    std::shared_ptr<Shared> shared;
    std::vector<std::thread> workers;

    static void work(std::shared_ptr<Shared> shared)
    {
        ShaderCompiler shaders;

        for (;;)
        {
            std::shared_ptr<KernelEntry> entry;
            {
                std::unique_lock<std::mutex> lock(shared->mutex);
                shared->wake.wait(
                    lock,
                    [&]
                    {
                        return shared->stopping || !shared->queue.empty();
                    }
                );
                if (shared->stopping)
                    return;
                entry = std::move(shared->queue.front());
                shared->queue.pop_front();
                shared->busy++;
            }

            std::string glsl;
            glsl.swap(entry->glsl);
            entry->compileError = createPipeline(shared->device, shared->cache, shared->pushConstants, shaders, *entry, glsl);

            bool stopping;
            {
                std::lock_guard<std::mutex> lock(shared->mutex);
                shared->busy--;
                stopping = shared->stopping;
                uint32_t old = entry->flags.fetch_or(KernelEntry::kReady, std::memory_order_acq_rel);
                if ((old & KernelEntry::kAbandoned) || stopping)
                    destroyPipeline(shared->device, *entry);
            }
            shared->done.notify_all();

            if (stopping)
                return;
        }
    }
};

// memory that buffers share, since allocating memory takes longer than a run on a buffer made for it; a block is freed
// with the last buffer in it
struct MemoryBlock
{
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint32_t type = 0;
    uint8_t* mapped = nullptr;
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> free; // offset and size of the free ranges, by offset
    uint32_t buffers = 0;
};

struct Memory
{
    MemoryBlock* block = nullptr;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;

    explicit operator bool() const
    {
        return block != nullptr;
    }
};

// device memory for a Luau buffer and the host memory mirroring it
struct Allocation
{
    VkDeviceSize deviceSize = 0;

    VkBuffer device = VK_NULL_HANDLE;
    Memory deviceMemory;
    VkBuffer staging = VK_NULL_HANDLE;
    Memory stagingMemory;
    void* mapped = nullptr;
};

// device copy of a Luau buffer, which stays on the device between runs
struct BufferEntry : Allocation
{
    size_t size = 0;

    // the device holds the contents of the staging memory, which the Luau buffer had when they last matched
    bool valid = false;
    uint64_t lastUsed = 0;

    // position in the least recently used list
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

// a buffer used by a run, once however many times it's listed
struct Use
{
    BufferEntry* entry;
    void* data;
    // its words in the binding of written parts
    uint32_t written = 0;
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

    struct HostWords
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        uint32_t words = 0;
    };
    HostWords status;
    // the parts of the buffers of a run that its kernel wrote
    HostWords written;
    // a copy of them the host reads, in cached memory: reading device memory from the host is slow
    HostWords writtenCopy;

    std::vector<std::unique_ptr<MemoryBlock>> blocks;

    HostThreads threads;

    ShaderCompiler shaders;
    std::function<CodeGen::GpuKernel(lua_State*, int)> lowering = CodeGen::compileGpu;

    std::string deviceName;
    RunStats stats;
    uint64_t runCounter = 0;

    // set when the device stopped responding or was lost; every run then fails with it
    std::string disabledReason;
    // set when the device can't run kernels with the VM's float semantics
    std::string floatReason;
    bool deviceBusy = false;

    // kernels are shared by every function lowering to the same shader; rejected functions share one per reason
    std::unordered_map<std::string, std::shared_ptr<KernelEntry>> kernels;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    std::unique_ptr<PipelineCompiler> compiler;

    std::unordered_map<uint64_t, BufferEntry> buffers;
    std::list<uint64_t> lru;
    uint64_t nextBufferId = 0;
    uint64_t cachedBytes = 0;


    // buffers and functions collected by the GC; what they hold is freed on the next call into the backend, since the GC
    // may run during one
    std::mutex deadMutex;
    std::vector<uint64_t> deadBuffers;
    std::vector<KernelEntry*> deadKernels;

    // reused by every run: the buffers in the order of the list, and once each
    std::vector<BufferEntry*> listed;
    std::vector<Use> uses;

    mutable std::mutex mutex;

    ~Impl();

    bool init(std::string& error);
    bool createHostWords(uint32_t words, HostWords& result, bool cached = false);
    void destroyHostWords(HostWords& words);
    std::string probe();
    void savePipelineCache();
    int findMemory(uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred);
    bool allocate(const VkMemoryRequirements& req, uint32_t type, bool map, Memory& mem);
    void release(Memory& mem);
    bool createBuffer(VkDeviceSize size, bool deviceLocal, VkBuffer& buffer, Memory& mem, void** mapped);
    void destroyAllocation(Allocation& alloc);
    void destroyBuffer(BufferEntry& entry);

    void destroyPipeline(KernelEntry& entry);
    bool evictLeastRecent();
    void freeDeadBuffers();

    std::string buildPipeline(KernelEntry& entry, const std::string& glsl);
    std::string checkLimits(const KernelEntry& entry);
    KernelEntry* kernelFor(lua_State* L, int fn);
    KernelEntry* exactFor(KernelEntry& kernel);
    BufferEntry* bufferFor(lua_State* L, int idx, bool create);
    Use* useFor(BufferEntry* entry, void* data);

    std::string run(lua_State* L, int fn, int list, uint32_t runs);
    std::string prepare(lua_State* L, const KernelEntry& kernel, int list, uint32_t runs);
    std::string download();
    void record(const KernelEntry& kernel, uint32_t runs);
    std::string submit();
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
    // 1.1 to query the float controls of the device, which shaders need to keep NaN, infinities and signed zeros
    app.apiVersion = VK_API_VERSION_1_1;

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

    // shaders declare SignedZeroInfNanPreserve for 32-bit floats; without it, kernels can't compute what the VM does
    std::vector<const char*> extensions;
    {
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> available(extensionCount);
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &extensionCount, available.data());

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical, &props);

        bool core = props.apiVersion >= VK_API_VERSION_1_2;
        bool listed = false;
        for (const VkExtensionProperties& e : available)
            listed |= strcmp(e.extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME) == 0;

        VkPhysicalDeviceFloatControlsPropertiesKHR floatControls = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &floatControls;
        if (core || listed)
            vkGetPhysicalDeviceProperties2(physical, &props2);

        if (!core && listed)
            extensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
        if (!(core || listed) || !floatControls.shaderSignedZeroInfNanPreserveFloat32)
            floatReason = "the device can't keep NaN, infinities and signed zeros in shaders";
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(extensions.size());
    dci.ppEnabledExtensionNames = extensions.data();
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

    // drivers check that the data is theirs, for this device and driver version, and ignore it otherwise
    std::vector<char> cached;
    if (!options.pipelineCachePath.empty())
    {
        if (FILE* f = fopen(options.pipelineCachePath.c_str(), "rb"))
        {
            char chunk[65536];
            for (size_t n; (n = fread(chunk, 1, sizeof(chunk), f)) > 0;)
                cached.insert(cached.end(), chunk, chunk + n);
            fclose(f);
        }
    }
    VkPipelineCacheCreateInfo pcci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pcci.initialDataSize = cached.size();
    pcci.pInitialData = cached.empty() ? nullptr : cached.data();
    if (vkCreatePipelineCache(device, &pcci, nullptr, &pipelineCache) != VK_SUCCESS)
    {
        pcci.initialDataSize = 0;
        pcci.pInitialData = nullptr;
        VK_CHECK(vkCreatePipelineCache(device, &pcci, nullptr, &pipelineCache));
    }

    if (!createHostWords(3, status))
    {
        error = "no host visible memory for the status words";
        return false;
    }

    // feature bits aren't enough: kernels must compute what the VM does, and runaway ones must stop when asked
    if (!floatReason.empty())
    {
        error = deviceName + " is refused: " + floatReason;
        return false;
    }

    std::string refusal = probe();
    if (!refusal.empty())
    {
        error = deviceName + " is refused: " + refusal;
        return false;
    }

    return true;
}

#undef VK_CHECK

// the status words are read by every invocation and written by the host while a run is going on, and kernels read and set
// the words of written parts, so both live in device memory the host can map when there is some (resizable BAR), and in
// host memory otherwise
bool VulkanBackend::Impl::createHostWords(uint32_t words, HostWords& result, bool cached)
{
    HostWords w;
    w.words = words;

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = words * 4;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(device, &bci, nullptr, &w.buffer) != VK_SUCCESS)
        return false;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, w.buffer, &req);
    int type = findMemory(
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        cached ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = uint32_t(type);
    if (type < 0 || vkAllocateMemory(device, &mai, nullptr, &w.memory) != VK_SUCCESS ||
        vkBindBufferMemory(device, w.buffer, w.memory, 0) != VK_SUCCESS || vkMapMemory(device, w.memory, 0, bci.size, 0, &w.mapped) != VK_SUCCESS)
    {
        destroyHostWords(w);
        return false;
    }

    result = w;
    return true;
}

void VulkanBackend::Impl::destroyHostWords(HostWords& words)
{
    vkDestroyBuffer(device, words.buffer, nullptr);
    vkFreeMemory(device, words.memory, nullptr);
    words = HostWords();
}

// runs a kernel that checks what the shaders the lowering produces rely on: NaN, infinities and signed zeros are kept, and
// a loop sees the stop word the host writes while it runs
static const char* const kProbe = R"(#version 450
#extension GL_EXT_spirv_intrinsics : require
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4466], 4461, 32);
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"], capabilities = [4464], 4459, 32);
layout(local_size_x = 1) in;
layout(std430, set = 0, binding = 0) buffer B0 { uint d[]; } b0;
layout(std430, set = 0, binding = 1) coherent volatile buffer BS { uint err; int stop; uint started; } st;
layout(push_constant) uniform PC { uint n; uint budget; uint len[1]; uint written; } pc;

void main()
{
    st.started = 1u;
    float z = uintBitsToFloat(b0.d[0]), h = uintBitsToFloat(b0.d[1]);
    b0.d[2] = floatBitsToUint(h - h);
    b0.d[3] = floatBitsToUint(z / z);
    b0.d[4] = floatBitsToUint(z * -1.0);
    b0.d[5] = floatBitsToUint(1.0 / (z * -1.0));
    b0.d[7] = floatBitsToUint(uintBitsToFloat(b0.d[1] - 0x7f000000u) * 0.5);
    b0.d[8] = floatBitsToUint(uintBitsToFloat(b0.d[1] - 0x7f7fffffu) + uintBitsToFloat(b0.d[1] - 0x7f7ffffeu));
    uint i = 0u;
    for (; i < 0x40000000u; i++)
    {
        if ((i & 255u) == 0u && st.stop != 0)
            break;
    }
    b0.d[6] = i < 0x40000000u ? 1u : 0u;
}
)";

std::string VulkanBackend::Impl::probe()
{
    KernelEntry kernel;
    kernel.name = "probe";
    kernel.written = {true};
    std::string reason = buildPipeline(kernel, kProbe);
    if (!reason.empty())
        return "the probe kernel failed to build: " + reason;

    BufferEntry buffer;
    buffer.deviceSize = 64;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindings};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(device, &dpci, nullptr, &pool) != VK_SUCCESS)
        pool = VK_NULL_HANDLE;

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &kernel.setLayout;

    if (!pool || vkAllocateDescriptorSets(device, &dsai, &set) != VK_SUCCESS ||
        !createBuffer(64, true, buffer.device, buffer.deviceMemory, nullptr) ||
        !createBuffer(64, false, buffer.staging, buffer.stagingMemory, &buffer.mapped))
        reason = "not enough memory for the probe kernel";

    uint32_t* words = static_cast<uint32_t*>(buffer.mapped);
    if (reason.empty())
    {
        memset(words, 0, 64);
        words[1] = 0x7f800000; // infinity; the kernel reads its inputs, so the compiler can't fold them

        VkDescriptorBufferInfo infos[2] = {{buffer.device, 0, VK_WHOLE_SIZE}, {status.buffer, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet writes[2] = {};
        for (uint32_t i = 0; i < 2; ++i)
        {
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(commandBuffer, 0);
        vkBeginCommandBuffer(commandBuffer, &begin);
        copyBuffer(commandBuffer, buffer.staging, buffer.device, 64);
        barrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        );
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipelineLayout, 0, 1, &set, 0, nullptr);
        PushConstants push = {};
        vkCmdPushConstants(commandBuffer, kernel.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
        barrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT
        );
        copyBuffer(commandBuffer, buffer.device, buffer.staging, 64);
        barrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        vkEndCommandBuffer(commandBuffer);

        // the host asks the kernel to stop as soon as it runs; the probe isn't bound by the time limit of runs
        VulkanOptions saved = options;
        options.abortMs = 0;
        options.timeoutMs = saved.timeoutMs > 5000 ? saved.timeoutMs : 5000;
        reason = submit();
        options = saved;
    }

    auto nan = [](uint32_t w)
    {
        return (w & 0x7f800000) == 0x7f800000 && (w & 0x007fffff) != 0;
    };
    if (reason.empty())
    {
        // half the smallest normal number is denormal
        if (!nan(words[2]) || !nan(words[3]) || words[4] != 0x80000000 || words[5] != 0xff800000 || words[7] != 0x00400000 || words[8] != 0x00000003)
            reason = "its shaders don't keep NaN, infinities, signed zeros and denormal numbers";
        if (words[6] != 1)
            reason += std::string(reason.empty() ? "its" : ", and its") + " shaders don't see the host asking a running kernel to stop";
    }

    if (!deviceBusy)
    {
        vkDestroyDescriptorPool(device, pool, nullptr);
        destroyBuffer(buffer);
        destroyPipeline(kernel);
    }
    return reason;
}

VulkanBackend::Impl::~Impl()
{
    // a driver still creating a pipeline can't be interrupted: its thread is left to finish, so the device must outlive it
    if (compiler && compiler->shutdown())
        deviceBusy = true;

    // with the pipelines made so far, even if one is still being made
    if (device)
        savePipelineCache();

    // a device that may still be executing a run can't be waited for or have its resources destroyed; leak them instead
    if (device && !deviceBusy)
    {
        vkDeviceWaitIdle(device);

        for (auto& [key, entry] : kernels)
        {
            if (entry->ready())
                destroyPipeline(*entry);
            if (entry->exact && entry->exact->ready())
                destroyPipeline(*entry->exact);
        }
        vkDestroyPipelineCache(device, pipelineCache, nullptr);
        for (auto& [id, entry] : buffers)
            destroyBuffer(entry);

        destroyHostWords(status);
        destroyHostWords(written);
        destroyHostWords(writtenCopy);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
    }

    if (instance && !deviceBusy)
        vkDestroyInstance(instance, nullptr);
}

void VulkanBackend::Impl::savePipelineCache()
{
    if (options.pipelineCachePath.empty() || !pipelineCache)
        return;

    size_t size = 0;
    if (vkGetPipelineCacheData(device, pipelineCache, &size, nullptr) != VK_SUCCESS || size == 0)
        return;
    std::vector<char> data(size);
    if (vkGetPipelineCacheData(device, pipelineCache, &size, data.data()) != VK_SUCCESS)
        return;

    // written next to it first, so that a process stopped meanwhile doesn't leave a truncated file
    std::string temporary = options.pipelineCachePath + ".tmp";
    FILE* f = fopen(temporary.c_str(), "wb");
    if (!f)
        return;
    bool ok = fwrite(data.data(), 1, size, f) == size;
    ok = fclose(f) == 0 && ok;
    if (ok)
        rename(temporary.c_str(), options.pipelineCachePath.c_str());
    else
        remove(temporary.c_str());
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

// blocks grow with the memory in use, up to a size large enough for many buffers of a typical run; larger buffers get
// a block of their own
constexpr VkDeviceSize kMinBlockSize = 1 << 20;
constexpr VkDeviceSize kBlockSize = 64 << 20;

bool VulkanBackend::Impl::allocate(const VkMemoryRequirements& req, uint32_t type, bool map, Memory& mem)
{
    auto take = [&](MemoryBlock& block)
    {
        for (size_t i = 0; i < block.free.size(); ++i)
        {
            auto [offset, size] = block.free[i];
            VkDeviceSize start = (offset + req.alignment - 1) / req.alignment * req.alignment;
            if (start + req.size > offset + size)
                continue;

            // what's left before and after the buffer stays free
            VkDeviceSize end = start + req.size;
            block.free.erase(block.free.begin() + i);
            if (end < offset + size)
                block.free.insert(block.free.begin() + i, {end, offset + size - end});
            if (start > offset)
                block.free.insert(block.free.begin() + i, {offset, start - offset});

            block.buffers++;
            mem = {&block, start, req.size};
            return true;
        }
        return false;
    };

    // device copies and host mirrors share a memory type on devices with unified memory, so blocks are mapped when needed
    for (std::unique_ptr<MemoryBlock>& block : blocks)
    {
        if (block->type != type)
            continue;

        void* mapped = nullptr;
        if (map && !block->mapped && vkMapMemory(device, block->memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
            continue;
        if (mapped)
            block->mapped = static_cast<uint8_t*>(mapped);

        if (take(*block))
            return true;
    }

    VkDeviceSize used = 0;
    for (std::unique_ptr<MemoryBlock>& block : blocks)
        used += block->type == type ? block->size : 0;

    VkDeviceSize size = kMinBlockSize;
    while (size < kBlockSize && (size < req.size * 2 || size < used))
        size *= 2;

    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size > kBlockSize / 2 ? req.size : size;
    mai.memoryTypeIndex = type;

    std::unique_ptr<MemoryBlock> block = std::make_unique<MemoryBlock>();
    block->type = type;
    block->size = mai.allocationSize;
    block->free = {{0, mai.allocationSize}};
    if (vkAllocateMemory(device, &mai, nullptr, &block->memory) != VK_SUCCESS)
        return false;
    if (map && vkMapMemory(device, block->memory, 0, VK_WHOLE_SIZE, 0, (void**)&block->mapped) != VK_SUCCESS)
    {
        vkFreeMemory(device, block->memory, nullptr);
        return false;
    }

    blocks.push_back(std::move(block));
    return take(*blocks.back());
}

void VulkanBackend::Impl::release(Memory& mem)
{
    MemoryBlock* block = mem.block;
    if (!block)
        return;

    // back into the free ranges, merged with its neighbors
    auto& free = block->free;
    auto it = std::lower_bound(free.begin(), free.end(), std::make_pair(mem.offset, VkDeviceSize(0)));
    it = free.insert(it, {mem.offset, mem.size});
    if (it + 1 != free.end() && it->first + it->second == (it + 1)->first)
    {
        it->second += (it + 1)->second;
        free.erase(it + 1);
    }
    if (it != free.begin() && (it - 1)->first + (it - 1)->second == it->first)
    {
        (it - 1)->second += it->second;
        free.erase(it);
    }

    mem = Memory();
    if (--block->buffers > 0)
        return;

    vkFreeMemory(device, block->memory, nullptr);
    for (size_t i = 0; i < blocks.size(); ++i)
    {
        if (blocks[i].get() == block)
        {
            blocks.erase(blocks.begin() + i);
            break;
        }
    }
}

// device-local buffers hold the copies kernels use; host buffers are mapped and cached, since they are compared and read
bool VulkanBackend::Impl::createBuffer(VkDeviceSize size, bool deviceLocal, VkBuffer& buffer, Memory& mem, void** mapped)
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

    if (!allocate(req, uint32_t(type), mapped != nullptr, mem))
        return false;

    if (deviceLocal)
        cachedBytes += size;

    if (mapped)
        *mapped = mem.block->mapped + mem.offset;

    return vkBindBufferMemory(device, buffer, mem.block->memory, mem.offset) == VK_SUCCESS;
}

void VulkanBackend::Impl::destroyAllocation(Allocation& alloc)
{
    if (alloc.deviceMemory)
        cachedBytes -= alloc.deviceSize;
    vkDestroyBuffer(device, alloc.device, nullptr);
    release(alloc.deviceMemory);
    vkDestroyBuffer(device, alloc.staging, nullptr);
    release(alloc.stagingMemory);
    alloc = Allocation();
}

void VulkanBackend::Impl::destroyBuffer(BufferEntry& entry)
{
    if (entry.listed)
        lru.erase(entry.lru);
    destroyAllocation(entry);
    entry = BufferEntry();
}

void VulkanBackend::Impl::destroyPipeline(KernelEntry& entry)
{
    Luau::Gpu::destroyPipeline(device, entry);
}

// frees the least recently used copy that the current run doesn't use, which the Luau buffer has the contents of; buffers
// are moved to the back of the list when used, so the ones the current run uses are at the back
bool VulkanBackend::Impl::evictLeastRecent()
{
    while (!lru.empty())
    {
        BufferEntry& e = buffers[lru.front()];
        if (e.lastUsed == runCounter)
            return false;

        lru.pop_front();
        e.listed = false;

        if (!e.deviceMemory)
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

        for (KernelEntry* k : {kernel, kernel->exact.get()})
        {
            if (!k)
                continue;
            uint32_t old = k->flags.fetch_or(KernelEntry::kAbandoned, std::memory_order_acq_rel);
            if ((old & KernelEntry::kReady) && !deviceBusy)
                destroyPipeline(*k);
        }
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
    std::string reason = checkLimits(entry);
    if (reason.empty())
        reason = createPipeline(device, pipelineCache, uint32_t(sizeof(PushConstants)), shaders, entry, glsl);
    entry.flags.fetch_or(KernelEntry::kReady);
    return reason;
}

std::string VulkanBackend::Impl::checkLimits(const KernelEntry& entry)
{
    uint32_t bindings = uint32_t(entry.written.size()) + 2;
    if (bindings > limits.storageBuffers)
        return "kernel uses " + std::to_string(bindings) +
               " storage buffers (its buffer, the status words and the written parts), the device supports " + std::to_string(limits.storageBuffers);
    if (sizeof(PushConstants) > limits.pushConstants)
        return "device doesn't support " + std::to_string(sizeof(PushConstants)) + " bytes of push constants";
    return "";
}

KernelEntry* VulkanBackend::Impl::exactFor(KernelEntry& kernel)
{
    if (!kernel.exact)
    {
        kernel.exact = std::make_shared<KernelEntry>();
        kernel.exact->name = kernel.name;
        kernel.exact->written = kernel.written;
        kernel.exact->glsl = std::move(kernel.exactGlsl);
        compiler->submit(kernel.exact);
    }
    return kernel.exact.get();
}

// the kernel of a function, lowered on its first run; functions lowering to the same shader share its pipeline
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
    std::shared_ptr<KernelEntry>& entry = kernels[key];

    if (!entry)
    {
        entry = std::make_shared<KernelEntry>();
        entry->key = key;
        entry->name = kernel.name;
        entry->written = kernel.bufferWritten;
        entry->reason = kernel.reason.empty() ? checkLimits(*entry) : kernel.reason;

        // drivers take a time growing faster than the code, about 40 s for 600 KB of GLSL and 160 s for 1.1 MB on NVIDIA's;
        // larger kernels would keep a compile thread for most of an hour
        if (entry->reason.empty() && kernel.glsl.size() > kMaxKernelGlsl)
            entry->reason = "a kernel of " + std::to_string(kernel.glsl.size() >> 10) + " KB of GLSL, beyond the " +
                            std::to_string(kMaxKernelGlsl >> 10) + " KB that drivers compile in a few minutes,";

        if (entry->reason.empty())
        {
            if (!compiler)
                compiler = std::make_unique<PipelineCompiler>(device, pipelineCache, uint32_t(sizeof(PushConstants)));
            entry->exactGlsl = kernel.exactGlsl();
            entry->glsl = std::move(kernel.glsl);
            compiler->submit(entry);
        }
        else
            entry->flags.fetch_or(KernelEntry::kReady);
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

    entry.lastUsed = runCounter;
    if (entry.listed)
        lru.splice(lru.end(), lru, entry.lru);
    else
        entry.lru = lru.insert(lru.end(), id);
    entry.listed = true;

    if (entry.deviceMemory)
        return &entry;

    entry.size = size;
    entry.valid = false;

    VkDeviceSize deviceSize = size == 0 ? 4 : (size + 3) & ~size_t(3);

    // copies of collected buffers are freed with them; the budget only bounds copies of live buffers. If allocation still
    // fails, free every copy this run doesn't need and retry
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        if (attempt > 0 || cachedBytes + deviceSize > options.cacheBytes)
            freeDeadBuffers();

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

    uses.push_back({entry, data});
    return &uses.back();
}

const char* const kCannotRun = "Function cannot run in GPU";

// buffers are compared with their device copies, uploaded, and downloaded when kernels write them, in parts of this size
constexpr VkDeviceSize kChunk = VkDeviceSize(1) << kWrittenChunkShift;

std::string VulkanBackend::Impl::run(lua_State* L, int fn, int list, uint32_t runs)
{
    stats.runs++;
    runCounter++;

    if (!disabledReason.empty())
        return disabledReason;

    // kernels are lowered once, but the builtins they use can be replaced afterwards
    KernelEntry* kernel = kernelFor(L, fn);
    if (!kernel->ready())
    {
        if (!options.waitForPipelines)
            return kCompiling;
        compiler->wait(*kernel);
    }
    if (kernel->reason.empty() && !kernel->compileError.empty())
        kernel->reason = kernel->compileError;
    if (!kernel->reason.empty() || !CodeGen::isGpuEnvironmentSafe(L, fn))
        return kCannotRun;

    // the kernel's exact variant runs instead when it needs it, which a run that needed it says. Its runs are discarded
    // like failed ones: the buffers are uploaded again for the variant
    KernelEntry* used = kernel;
    if (kernel->preferExact > 0 && kernel->exact && kernel->exact->ready() && kernel->exact->compileError.empty())
    {
        kernel->preferExact--;
        used = kernel->exact.get();
    }

    std::string reason;
    uint32_t failure = 0;
    for (;;)
    {
        reason = prepare(L, *used, list, runs);
        if (!reason.empty() || runs == 0 || listed.empty())
            return reason;

        record(*used, runs);
        reason = submit();

        // until the device copies are uploaded again, they can't be trusted
        failure = *(volatile uint32_t*)status.mapped;
        if (!reason.empty() || failure != 0)
        {
            for (Use& u : uses)
                u.entry->valid = false;
        }
        if (!reason.empty() || (failure >> 24) != uint32_t(CodeGen::GpuFailure::NeedsExact) || used != kernel)
            break;

        KernelEntry* exact = exactFor(*kernel);
        if (!exact->ready())
        {
            if (!options.waitForPipelines)
                return kCompiling;
            compiler->wait(*exact);
        }
        if (!exact->compileError.empty())
            return kCannotRun;
        kernel->preferExact = 64;
        stats.checkedRuns++;
        used = exact;
    }

    if (!reason.empty())
        return reason;
    if (failure != 0)
    {
        stats.failures++;
        return CodeGen::describeGpuFailure(failure);
    }

    // the parts the kernel didn't write are still those the Luau buffers have, since they were uploaded or found the same;
    // the parts it wrote are downloaded
    if (used->written[0])
    {
        reason = download();
        if (!reason.empty())
        {
            for (Use& u : uses)
                u.entry->valid = false;
            return reason;
        }
    }

    stats.succeeded++;
    return "";
}

std::string VulkanBackend::Impl::download()
{
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &begin);

    // what the run wrote, in the previous submission
    barrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT
    );

    const uint32_t* flags = static_cast<const uint32_t*>(writtenCopy.mapped);
    std::vector<VkBufferCopy> regions;
    std::vector<std::vector<VkBufferCopy>> copied(uses.size());
    bool submitting = false;
    for (size_t i = 0; i < uses.size(); ++i)
    {
        BufferEntry* e = uses[i].entry;
        regions.clear();

        for (VkDeviceSize offset = 0; offset < e->deviceSize; offset += kChunk)
        {
            if (flags[uses[i].written + offset / kChunk] == 0)
                continue;

            VkDeviceSize size = std::min(kChunk, e->deviceSize - offset);
            if (!regions.empty() && regions.back().srcOffset + regions.back().size == offset)
                regions.back().size += size;
            else
                regions.push_back({offset, offset, size});
        }

        // a single part is already in the host mirror
        if (!regions.empty() && e->deviceSize > kChunk)
        {
            vkCmdCopyBuffer(commandBuffer, e->device, e->staging, uint32_t(regions.size()), regions.data());
            submitting = true;
        }
        copied[i] = regions;
    }

    barrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    vkEndCommandBuffer(commandBuffer);

    std::string reason = submitting ? submit() : "";
    if (!reason.empty())
        return reason;

    // copied back to the Luau buffers in parts, by several threads
    std::vector<std::pair<size_t, VkDeviceSize>> parts;
    for (size_t i = 0; i < uses.size(); ++i)
    {
        BufferEntry* e = uses[i].entry;
        for (const VkBufferCopy& r : copied[i])
        {
            for (VkDeviceSize offset = r.srcOffset; offset < r.srcOffset + r.size; offset += kChunk)
                parts.push_back({i, offset});
            // the last word may be partly past the end of the Luau buffer
            stats.downloadedBytes += std::min<VkDeviceSize>(r.size, e->size - std::min<VkDeviceSize>(r.srcOffset, e->size));
        }
    }

    threads.run(
        parts.size(),
        [&](size_t i)
        {
            const Use& u = uses[parts[i].first];
            VkDeviceSize offset = parts[i].second;
            size_t bytes = size_t(std::min<VkDeviceSize>(kChunk, u.entry->size - std::min<VkDeviceSize>(offset, u.entry->size)));
            memcpy(static_cast<uint8_t*>(u.data) + offset, static_cast<uint8_t*>(u.entry->mapped) + offset, bytes);
        }
    );

    return "";
}

std::string VulkanBackend::Impl::prepare(lua_State* L, const KernelEntry& kernel, int list, uint32_t runs)
{
    uint64_t groups = (uint64_t(runs) + kWorkgroupSize - 1) / kWorkgroupSize;
    uint64_t groupsX = groups < limits.groupsX ? groups : limits.groupsX;
    if (groupsX > 0 && (groups + groupsX - 1) / groupsX > limits.groupsY)
        return std::to_string(runs) + " runs per buffer exceed the device's workgroup limits";

    listed.clear();
    uses.clear();

    for (int i = 1, n = lua_objlen(L, list); i <= n; ++i)
    {
        lua_rawgeti(L, list, i);
        size_t len = 0;
        void* data = lua_tobuffer(L, -1, &len);

        if (((uint64_t(len) + 3) & ~uint64_t(3)) > limits.storageRange)
        {
            lua_pop(L, 1);
            return "a buffer of " + std::to_string(len) + " bytes exceeds the device's storage buffer range of " +
                   std::to_string(limits.storageRange) + " bytes";
        }

        BufferEntry* entry = bufferFor(L, -1, true);
        lua_pop(L, 1);
        if (!entry)
            return "not enough device memory for a buffer of " + std::to_string(len) + " bytes";

        listed.push_back(entry);
        useFor(entry, data);
    }

    // a word per part of each buffer, for the kernel to set when it writes there
    uint32_t words = 0;
    for (Use& u : uses)
    {
        u.written = words;
        words += uint32_t((u.entry->deviceSize + kChunk - 1) / kChunk);
    }
    if (words > written.words)
    {
        destroyHostWords(written);
        destroyHostWords(writtenCopy);
        uint32_t size = std::max(words, written.words * 2);
        if (!createHostWords(size, written) || !createHostWords(size, writtenCopy, true))
            return "not enough memory to track what a run writes";
    }

    if (descriptorSets < listed.size())
    {
        if (descriptorPool)
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        uint32_t sets = descriptorSets ? descriptorSets : 16;
        while (sets < listed.size())
            sets *= 2;

        VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sets * kBindings};
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

void VulkanBackend::Impl::record(const KernelEntry& kernel, uint32_t runs)
{
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &begin);

    // only the parts of a buffer that differ from its device copy are uploaded; they're compared and copied by several threads
    std::vector<std::pair<Use*, VkDeviceSize>> parts;
    for (Use& u : uses)
    {
        for (VkDeviceSize offset = 0; offset < u.entry->deviceSize; offset += kChunk)
            parts.push_back({&u, offset});
    }

    std::vector<uint8_t> changed(parts.size(), 0);
    threads.run(
        parts.size(),
        [&](size_t i)
        {
            BufferEntry* e = parts[i].first->entry;
            VkDeviceSize offset = parts[i].second;
            // the last word may be partly past the end of the Luau buffer
            size_t bytes =
                size_t(std::min<VkDeviceSize>(std::min(kChunk, e->deviceSize - offset), e->size - std::min<VkDeviceSize>(offset, e->size)));
            uint8_t* mirror = static_cast<uint8_t*>(e->mapped) + offset;
            const uint8_t* data = static_cast<const uint8_t*>(parts[i].first->data) + offset;
            if (e->valid && memcmp(mirror, data, bytes) == 0)
                return;

            memcpy(mirror, data, bytes);
            changed[i] = 1;
        }
    );

    std::vector<VkBufferCopy> regions;
    for (size_t i = 0; i < parts.size();)
    {
        BufferEntry* e = parts[i].first->entry;
        regions.clear();
        for (; i < parts.size() && parts[i].first->entry == e; ++i)
        {
            if (!changed[i])
                continue;

            VkDeviceSize offset = parts[i].second, size = std::min(kChunk, e->deviceSize - offset);
            if (!regions.empty() && regions.back().srcOffset + regions.back().size == offset)
                regions.back().size += size;
            else
                regions.push_back({offset, offset, size});
            stats.uploadedBytes += std::min<VkDeviceSize>(size, e->size - std::min<VkDeviceSize>(offset, e->size));
        }

        if (!regions.empty())
            vkCmdCopyBuffer(commandBuffer, e->staging, e->device, uint32_t(regions.size()), regions.data());
        e->valid = true;
    }

    // no part is written yet
    vkCmdFillBuffer(commandBuffer, written.buffer, 0, VK_WHOLE_SIZE, 0);

    barrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    );

    vkResetDescriptorPool(device, descriptorPool, 0);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);

    uint32_t groups = uint32_t((uint64_t(runs) + kWorkgroupSize - 1) / kWorkgroupSize);
    uint32_t groupsX = groups < limits.groupsX ? groups : limits.groupsX;

    // buffers run concurrently, except that a buffer listed again runs after what it was listed for earlier
    for (size_t i = 0; i < listed.size(); ++i)
    {
        BufferEntry* entry = listed[i];
        if (std::find(listed.begin(), listed.begin() + i, entry) != listed.begin() + i)
            barrier(
                commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
            );

        VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = descriptorPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &kernel.setLayout;
        VkDescriptorSet set;
        vkAllocateDescriptorSets(device, &dsai, &set);

        VkDescriptorBufferInfo infos[kBindings] = {
            {entry->device, 0, VK_WHOLE_SIZE}, {status.buffer, 0, VK_WHOLE_SIZE}, {written.buffer, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet writes[kBindings] = {};
        for (uint32_t b = 0; b < kBindings; ++b)
        {
            writes[b] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[b].dstSet = set;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        }
        vkUpdateDescriptorSets(device, kBindings, writes, 0, nullptr);

        PushConstants push = {};
        push.count = runs;
        push.loopBudget = options.loopBudget;
        push.lengths[0] = uint32_t(entry->size);
        push.written = useFor(entry, nullptr)->written;

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(commandBuffer, kernel.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(commandBuffer, groupsX, (groups + groupsX - 1) / groupsX, 1);
    }

    // buffers of a single part come back at once, which spares a submission; the host then reads the status and written parts
    // words, and downloads the parts the kernel wrote of larger buffers
    if (kernel.written[0])
    {
        barrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT
        );
        for (Use& u : uses)
        {
            if (u.entry->deviceSize <= kChunk)
                copyBuffer(commandBuffer, u.entry->device, u.entry->staging, u.entry->deviceSize);
        }
        copyBuffer(commandBuffer, written.buffer, writtenCopy.buffer, VkDeviceSize(written.words) * 4);
    }

    barrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_HOST_READ_BIT
    );

    vkEndCommandBuffer(commandBuffer);
}

// runs the recorded commands; a timeout or a device error disables the backend
std::string VulkanBackend::Impl::submit()
{
    volatile uint32_t* words = static_cast<volatile uint32_t*>(status.mapped);
    words[0] = 0;
    words[1] = 0;
    words[2] = 0;

    VkSubmitInfo info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.commandBufferCount = 1;
    info.pCommandBuffers = &commandBuffer;
    vkResetFences(device, 1, &fence);

    // a run still going on abortMs after it started is asked to stop, which its loops check, and then completes with an
    // error status; time spent waiting for other work on the GPU doesn't count, which the third word tells
    using Clock = std::chrono::steady_clock;
    Clock::time_point submitted = Clock::now(), started = {};
    bool running = false;

    VkResult res = vkQueueSubmit(queue, 1, &info, fence);
    while (res == VK_SUCCESS)
    {
        res = vkWaitForFences(device, 1, &fence, VK_TRUE, 1000000);
        if (res != VK_TIMEOUT)
            break;

        Clock::time_point now = Clock::now();
        if (!running && words[2] != 0)
        {
            running = true;
            started = now;
        }

        if (running && words[1] == 0 && now - started > std::chrono::milliseconds(options.abortMs))
            words[1] = 1;

        if (now - submitted > std::chrono::milliseconds(options.timeoutMs))
            break;
        res = VK_SUCCESS;
    }

    if (res == VK_SUCCESS)
        return "";

    deviceBusy = res == VK_TIMEOUT;
    disabledReason = res == VK_TIMEOUT ? "the GPU backend was disabled after a run exceeded " + std::to_string(options.timeoutMs) + " ms"
                                       : "the GPU backend was disabled after a device error (" + std::to_string(res) + ")";
    return disabledReason;
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

void VulkanBackend::setLowering(std::function<CodeGen::GpuKernel(lua_State* L, int idx)> lowering)
{
    std::lock_guard<std::mutex> lock(impl->mutex);
    impl->lowering = std::move(lowering);
}

std::string VulkanBackend::run(lua_State* L, int fn, int list, uint32_t runs)
{
    fn = lua_absindex(L, fn);
    list = lua_absindex(L, list);

    std::lock_guard<std::mutex> lock(impl->mutex);
    std::string reason = impl->run(L, fn, list, runs);

    // after the run rather than before it, so that the memory blocks the buffers of an earlier run leave hold the buffers
    // made for this one instead of being freed and allocated again
    impl->freeDeadBuffers();
    return reason;
}

const std::string& VulkanBackend::deviceName() const
{
    return impl->deviceName;
}

RunStats VulkanBackend::stats() const
{
    std::lock_guard<std::mutex> lock(impl->mutex);
    RunStats s = impl->stats;
    s.cachedBytes = impl->cachedBytes;
    for (const std::unique_ptr<MemoryBlock>& block : impl->blocks)
        s.memoryBytes += block->size;
    s.kernels = impl->kernels.size();
    return s;
}

} // namespace Gpu
} // namespace Luau

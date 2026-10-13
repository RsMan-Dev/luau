// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "Luau/GpuVulkan.h"

#include "lua.h"
#include "lualib.h"

#include <shaderc/shaderc.h>
#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <unordered_map>

namespace Luau
{
namespace Gpu
{

namespace
{

constexpr uint32_t kMaxGroupsX = 65535;
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

} // namespace

struct VulkanBackend::Impl
{
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

    shaderc_compiler_t compiler = nullptr;

    std::string deviceName;
    DispatchStats stats;
    uint64_t dispatchCounter = 0;

    std::map<std::pair<std::string, int>, KernelEntry> kernels;
    std::unordered_map<const void*, BufferEntry> buffers;

    lua_GpuBackend hooks = {};
    std::function<void(const std::string&)> fallbackHandler;

    ~Impl();

    bool init(std::string& error);
    int findMemory(uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred);
    bool createBuffer(BufferEntry& entry, size_t size, bool deviceLocal);
    void destroyBuffer(BufferEntry& entry);
    BufferEntry* bufferFor(const void* data, size_t size);
    void evict();

    std::string buildPipeline(KernelEntry& entry, const Kernel& kernel);
    KernelEntry* lookup(lua_State* L, int fnidx);
    int dispatch(lua_State* L, int fnidx, int n, int argidx, int nargs);
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
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    // prefer discrete GPUs, then integrated ones; software rasterizers are a last resort
    int bestScore = -1;
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
            }
            break;
        }
    }

    if (!physical)
    {
        error = "no Vulkan device with a compute queue";
        return false;
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

    compiler = shaderc_compiler_initialize();
    if (!compiler)
    {
        error = "failed to initialize shaderc";
        return false;
    }

    hooks.context = this;
    hooks.dispatch = [](void* context, lua_State* L, int fnidx, int n, int argidx, int nargs)
    {
        return static_cast<Impl*>(context)->dispatch(L, fnidx, n, argidx, nargs);
    };
    hooks.status = [](void* context, lua_State* L, int fnidx)
    {
        Impl* self = static_cast<Impl*>(context);
        KernelEntry* entry = self->lookup(L, fnidx);
        if (!entry)
            lua_pushstring(L, "function is not marked @gpu or its chunk wasn't registered with the GPU backend");
        else if (!entry->reason.empty())
            lua_pushstring(L, entry->reason.c_str());
        else
            lua_pushnil(L);
    };

    return true;
}

#undef VK_CHECK

VulkanBackend::Impl::~Impl()
{
    if (device)
    {
        vkDeviceWaitIdle(device);

        for (auto& [key, entry] : kernels)
        {
            vkDestroyPipeline(device, entry.pipeline, nullptr);
            vkDestroyPipelineLayout(device, entry.pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, entry.setLayout, nullptr);
        }
        for (auto& [key, entry] : buffers)
            destroyBuffer(entry);

        vkDestroyBuffer(device, status.device, nullptr);
        vkFreeMemory(device, status.deviceMemory, nullptr);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
    }

    if (instance)
        vkDestroyInstance(instance, nullptr);

    if (compiler)
        shaderc_compiler_release(compiler);
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

bool VulkanBackend::Impl::createBuffer(BufferEntry& entry, size_t size, bool deviceLocal)
{
    VkBuffer& buffer = deviceLocal ? entry.device : entry.staging;
    VkDeviceMemory& mem = deviceLocal ? entry.deviceMemory : entry.stagingMemory;

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = entry.deviceSize;
    bci.usage = deviceLocal ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                            : VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(device, &bci, nullptr, &buffer) != VK_SUCCESS)
        return false;

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
        return false;
    if (vkBindBufferMemory(device, buffer, mem, 0) != VK_SUCCESS)
        return false;

    if (!deviceLocal && vkMapMemory(device, mem, 0, entry.deviceSize, 0, &entry.mapped) != VK_SUCCESS)
        return false;

    return true;
}

void VulkanBackend::Impl::destroyBuffer(BufferEntry& entry)
{
    if (entry.mapped)
        vkUnmapMemory(device, entry.stagingMemory);
    vkDestroyBuffer(device, entry.device, nullptr);
    vkFreeMemory(device, entry.deviceMemory, nullptr);
    vkDestroyBuffer(device, entry.staging, nullptr);
    vkFreeMemory(device, entry.stagingMemory, nullptr);
    entry = BufferEntry();
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
        entry.deviceSize = (size + 3) & ~size_t(3);
        if (entry.deviceSize == 0)
            entry.deviceSize = 4;

        if (!createBuffer(entry, size, true) || !createBuffer(entry, size, false))
        {
            destroyBuffer(entry);
            return nullptr;
        }

        it = buffers.emplace(data, entry).first;
    }

    it->second.lastUsed = dispatchCounter;
    return &it->second;
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
    shaderc_compile_options_t options = shaderc_compile_options_initialize();
    shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);
    shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

    shaderc_compilation_result_t result =
        shaderc_compile_into_spv(compiler, kernel.glsl.data(), kernel.glsl.size(), shaderc_compute_shader, kernel.name.c_str(), "main", options);
    shaderc_compile_options_release(options);

    if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success)
    {
        std::string message = std::string("internal error, generated shader doesn't compile: ") + shaderc_result_get_error_message(result);
        shaderc_result_release(result);
        return message;
    }

    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = shaderc_result_get_length(result);
    smci.pCode = reinterpret_cast<const uint32_t*>(shaderc_result_get_bytes(result));
    VkShaderModule module;
    VkResult res = vkCreateShaderModule(device, &smci, nullptr, &module);
    shaderc_result_release(result);
    if (res != VK_SUCCESS)
        return "vkCreateShaderModule failed";

    uint32_t bindings = uint32_t(kernel.bufferWritten.size()) + 1;
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
    vkCreateDescriptorSetLayout(device, &dslci, nullptr, &entry.setLayout);

    VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, uint32_t(sizeof(PushConstants))};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &entry.setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &range;
    vkCreatePipelineLayout(device, &plci, nullptr, &entry.pipelineLayout);

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.layout = entry.pipelineLayout;
    res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &entry.pipeline);
    vkDestroyShaderModule(device, module, nullptr);

    if (res != VK_SUCCESS)
        return "vkCreateComputePipelines failed";

    return "";
}

KernelEntry* VulkanBackend::Impl::lookup(lua_State* L, int fnidx)
{
    lua_pushvalue(L, fnidx);
    lua_Debug ar = {};
    int ok = lua_getinfo(L, -1, "s", &ar);
    lua_pop(L, 1);

    if (!ok || !ar.source || ar.linedefined <= 0)
        return nullptr;

    auto it = kernels.find({ar.source, ar.linedefined});
    return it == kernels.end() ? nullptr : &it->second;
}

int VulkanBackend::Impl::dispatch(lua_State* L, int fnidx, int n, int argidx, int nargs)
{
    stats.dispatches++;

    KernelEntry* kernel = lookup(L, fnidx);
    if (!kernel)
        return 0;

    if (!kernel->reason.empty())
    {
        // reported on first use rather than at load, since @gpu helpers aren't meant to be dispatched
        if (!kernel->warned && fallbackHandler)
            fallbackHandler("@gpu function '" + kernel->name + "' runs in the VM: " + kernel->reason);
        kernel->warned = true;
        return 0;
    }

    size_t nbuffers = kernel->written.size();
    if (size_t(nargs) < nbuffers)
        return 0; // missing buffers; the VM raises the error the script expects

    if (n == 0)
        return 1;

    dispatchCounter++;

    std::vector<BufferEntry*> entries(nbuffers);
    std::vector<void*> datas(nbuffers);
    uint32_t lengths[kMaxKernelBuffers] = {};

    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &begin);

    for (size_t i = 0; i < nbuffers; ++i)
    {
        size_t len = 0;
        void* data = lua_tobuffer(L, argidx + int(i), &len);
        BufferEntry* entry = bufferFor(data, len);
        if (!entry)
        {
            vkEndCommandBuffer(commandBuffer);
            return 0;
        }

        entries[i] = entry;
        datas[i] = data;
        lengths[i] = uint32_t(len);

        // the same buffer can be passed twice; upload it once
        bool seen = false;
        for (size_t j = 0; j < i; ++j)
            seen |= entries[j] == entry;
        if (seen)
            continue;

        if (!entry->valid || memcmp(entry->mapped, data, len) != 0)
        {
            memcpy(entry->mapped, data, len);
            VkBufferCopy copy = {0, 0, entry->deviceSize};
            vkCmdCopyBuffer(commandBuffer, entry->staging, entry->device, 1, &copy);
            entry->valid = true;
            stats.uploadedBytes += len;
        }
    }

    VkMemoryBarrier toCompute = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toCompute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    toCompute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &toCompute,
        0,
        nullptr,
        0,
        nullptr
    );

    vkResetDescriptorPool(device, descriptorPool, 0);
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descriptorPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &kernel->setLayout;
    VkDescriptorSet set;
    vkAllocateDescriptorSets(device, &dsai, &set);

    std::vector<VkDescriptorBufferInfo> infos(nbuffers + 1);
    std::vector<VkWriteDescriptorSet> writes(nbuffers + 1);
    for (size_t i = 0; i <= nbuffers; ++i)
    {
        infos[i].buffer = i < nbuffers ? entries[i]->device : status.device;
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;

        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = uint32_t(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, uint32_t(writes.size()), writes.data(), 0, nullptr);

    PushConstants push = {};
    push.count = uint32_t(n);
    memcpy(push.lengths, lengths, sizeof(lengths));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(commandBuffer, kernel->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t groups = (uint32_t(n) + kWorkgroupSize - 1) / kWorkgroupSize;
    uint32_t gx = groups < kMaxGroupsX ? groups : kMaxGroupsX;
    uint32_t gy = (groups + gx - 1) / gx;
    vkCmdDispatch(commandBuffer, gx, gy, 1);

    VkMemoryBarrier toTransfer = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toTransfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &toTransfer, 0, nullptr, 0, nullptr
    );

    for (size_t i = 0; i < nbuffers; ++i)
    {
        bool seen = false;
        for (size_t j = 0; j < i; ++j)
            seen |= entries[j] == entries[i] && kernel->written[j];
        if (!kernel->written[i] || seen)
            continue;

        VkBufferCopy copy = {0, 0, entries[i]->deviceSize};
        vkCmdCopyBuffer(commandBuffer, entries[i]->device, entries[i]->staging, 1, &copy);
    }

    VkMemoryBarrier toHost = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &toHost,
        0,
        nullptr,
        0,
        nullptr
    );

    vkEndCommandBuffer(commandBuffer);

    *(volatile uint32_t*)status.mapped = 0;

    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer;
    vkResetFences(device, 1, &fence);
    if (vkQueueSubmit(queue, 1, &submit, fence) != VK_SUCCESS || vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
    {
        for (BufferEntry* entry : entries)
            entry->valid = false;
        return 0;
    }

    // an error on the GPU (out of bounds or unaligned access, integer division by zero) discards the results;
    // the VM then reruns the whole dispatch and raises the error Luau would
    if (*(volatile uint32_t*)status.mapped != 0)
    {
        for (size_t i = 0; i < nbuffers; ++i)
        {
            if (kernel->written[i])
                entries[i]->valid = false;
        }
        stats.fallbacks++;
        return 0;
    }

    for (size_t i = 0; i < nbuffers; ++i)
    {
        if (!kernel->written[i])
            continue;
        memcpy(datas[i], entries[i]->mapped, entries[i]->size);
        stats.downloadedBytes += entries[i]->size;
    }

    stats.gpuDispatches++;

    if (dispatchCounter % 256 == 0)
        evict();

    return 1;
}

std::unique_ptr<VulkanBackend> VulkanBackend::create(std::string& error)
{
    std::unique_ptr<VulkanBackend> backend(new VulkanBackend());
    backend->impl = std::make_unique<Impl>();
    if (!backend->impl->init(error))
        return nullptr;
    return backend;
}

VulkanBackend::~VulkanBackend() = default;

std::vector<Kernel> VulkanBackend::load(const std::string& chunkname, const char* source, size_t size)
{
    std::vector<Kernel> kernels = translate(source, size);

    for (Kernel& kernel : kernels)
    {
        KernelEntry& entry = impl->kernels[{chunkname, kernel.line}];

        // reloading a chunk replaces its kernels
        vkDestroyPipeline(impl->device, entry.pipeline, nullptr);
        vkDestroyPipelineLayout(impl->device, entry.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(impl->device, entry.setLayout, nullptr);
        entry = KernelEntry();

        entry.name = kernel.name;
        entry.written = kernel.bufferWritten;
        if (!kernel.reason.empty())
            entry.reason = "line " + std::to_string(kernel.reasonLocation.begin.line + 1) + ": " + kernel.reason;

        if (entry.reason.empty())
            entry.reason = impl->buildPipeline(entry, kernel);

        if (!entry.reason.empty())
            kernel.reason = entry.reason;

        if (getenv("LUAU_GPU_DUMP"))
            fprintf(
                stderr,
                "// kernel %s (line %d)%s%s\n%s\n",
                kernel.name.c_str(),
                kernel.line,
                kernel.reason.empty() ? "" : ": ",
                kernel.reason.c_str(),
                kernel.glsl.c_str()
            );
    }

    return kernels;
}

void VulkanBackend::setFallbackHandler(std::function<void(const std::string& message)> handler)
{
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

DispatchStats& VulkanBackend::stats()
{
    return impl->stats;
}

} // namespace Gpu
} // namespace Luau

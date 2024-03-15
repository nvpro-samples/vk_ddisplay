/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "vkdd.hpp"

#include "buffer_allocation.hpp"
#include "canvas_region.hpp"
#include "image_allocation.hpp"
#include "triangle_mesh.hpp"
#include "triangle_mesh_instance_set.hpp"
#include "vulkan_memory_pool.hpp"

#include <unordered_set>

namespace vkdd {
struct GlobalData
{
  Mat4x4f m_view;
  Mat4x4f m_proj;
  float   m_runtimeMillis;
};

// vk_ddisplay
// a logical device represents a Vulkan device group and manages different things
// * a set of enabled logical displays attached to the device group's physical devices
// * a graphics queue for rendering and a dedicated transfer queue for host -> device / device -> device transfers
// * a set of buffered command execution units which provide an easy way to record multiple command buffers in parallel
// * a set of memory pools, one for each physical device
// * a single staging memory pool (host-visible and host coherent)
// * the main render pass and device local triangle mesh geomtry resources
class LogicalDevice
{
public:
  typedef std::unique_ptr<class LogicalDisplay> UniqueLogicalDisplay;

  LogicalDevice(vk::Instance instance, uint32_t devGroupIdx);
  ~LogicalDevice();

  [[nodiscard]] LogicalDisplay* enableDisplay(class Scene const& scene, vk::DisplayKHR display, CanvasRegion displayRegionOnCanvas);
  vk::Instance       vkInstance() const { return m_instance; }
  vk::Device         vkDevice() const { return m_device.get(); }
  FrameIndex         getCurrentFrameIndex() const { return m_frameIndex; }
  vk::PhysicalDevice getPhysicalDevice(DeviceIndex deviceIndex) const { return m_physicalDevices[deviceIndex]; }
  uint32_t           getNumPhysicalDevices() const { return (uint32_t)m_physicalDevices.size(); }
  uint32_t           getGraphicsQueueFamilyIndex() const { return m_graphicsQueueFamilyIndex; }
  uint32_t           getTransferQueueFamilyIndex() const { return m_transferQueueFamilyIndex; }
  vk::Queue          getQueue(uint32_t queueFamilyIndex) const;
  class VulkanMemoryObjectUploader& getUploader() const { return *m_uploader; }
  [[nodiscard]] bool                start();
  void                              render();
  void                              interrupt();
  void                              join();

  VulkanMemoryPool::Allocation allocateHostVisibleDeviceMemory(vk::MemoryRequirements memReqs,
                                                               void const*            initialData       = nullptr,
                                                               size_t                 initialDataLength = 0);
  VulkanMemoryPool::Allocation allocateStagingMemory(size_t size, size_t alignment);
  VulkanMemoryPool::Allocation allocateDeviceMemory(OptionalDeviceIndex     deviceIndex,
                                                    vk::MemoryRequirements  memReqs,
                                                    vk::MemoryPropertyFlags memPropFlags);

  BufferAllocation allocateStagingBuffer(vk::BufferCreateInfo createInfo);
  BufferAllocation allocateBuffer(OptionalDeviceIndex deviceIndex, vk::BufferCreateInfo createInfo, vk::MemoryPropertyFlags memPropFlags);
  ImageAllocation allocateImage(OptionalDeviceIndex deviceIndex, vk::ImageCreateInfo createInfo, vk::MemoryPropertyFlags memPropFlags);

  void scheduleForDeallocation(VulkanMemoryPool::Allocation allocation, uint32_t remainingFramesToKeepAlive = NUM_QUEUED_FRAMES);
  void scheduleForDeallocation(BufferAllocation allocation, uint32_t remainingFramesToKeepAlive = NUM_QUEUED_FRAMES);
  void scheduleForDeallocation(ImageAllocation allocation, uint32_t remainingFramesToKeepAlive = NUM_QUEUED_FRAMES);

  vk::RenderPass     getDonutRenderPass() const { return m_donutRenderPass.get(); }
  vk::PipelineLayout getDonutPipelineLayout() const { return m_donutPipelineLayout.get(); }
  vk::Pipeline       getDonutPipeline() const { return m_donutPipeline.get(); }
  TriangleMesh*      getDonutTriangleMesh(DeviceIndex deviceIndex, uint32_t baseNumTesselations);

private:
  typedef std::unique_ptr<class CommandExecutionUnit>              UniqueCommandExecutionUnit;
  typedef std::unique_ptr<VulkanMemoryPool>                        UniqueVulkanMemoryPool;
  typedef std::unordered_map<MemTypeIndex, UniqueVulkanMemoryPool> MemPoolCollection;
  typedef std::unique_ptr<class CommandExecutionUnit>              UniqueCommandExecutionUnit;

  vk::Instance                                              m_instance;
  uint32_t                                                  m_devGroupIdx;
  std::vector<vk::PhysicalDevice>                           m_physicalDevices;
  vk::UniqueDevice                                          m_device;
  uint32_t                                                  m_graphicsQueueFamilyIndex;
  uint32_t                                                  m_transferQueueFamilyIndex;
  uint32_t                                                  m_framebufferTransferQueueFamilyIndex;
  std::unordered_map<uint32_t, vk::Queue>                   m_queues;
  vk::UniqueSemaphore                                       m_transferQueueSyncSemaphore;
  std::array<UniqueCommandExecutionUnit, NUM_QUEUED_FRAMES> m_cmdExecUnits;
  UniqueVulkanMemoryPool                                    m_stagingMemPool;
  MemPoolCollection                                         m_globalMemPools;
  std::vector<MemPoolCollection>                            m_perSubDeviceMemPools;
  std::mutex                                                m_memPoolsMtx;
  std::vector<struct DeallocationContainer>                 m_deallocationQueue;
  std::mutex                                                m_deallocationQueueMtx;
  std::unique_ptr<VulkanMemoryObjectUploader>               m_uploader;
  std::vector<UniqueLogicalDisplay>                         m_logicalDisplays;
  FrameIndex                                                m_frameIndex;

  // donut rendering
  vk::UniquePipelineCache                                                                      m_donutPipelineCache;
  vk::UniqueShaderModule                                                                       m_donutVert;
  vk::UniqueShaderModule                                                                       m_donutFrag;
  vk::UniquePipelineLayout                                                                     m_donutPipelineLayout;
  vk::UniquePipeline                                                                           m_donutPipeline;
  vk::UniqueRenderPass                                                                         m_donutRenderPass;
  std::unordered_map<DeviceIndex, std::unordered_map<uint32_t, std::unique_ptr<TriangleMesh>>> m_donutTriMeshes;
  std::mutex                                                                                   m_donutTriMeshesMtx;

  MemTypeIndex getMemoryTypeIndex(DeviceIndex deviceIndex, uint32_t memoryTypeBits, vk::MemoryPropertyFlags memPropFlags);
  VulkanMemoryPool* getMemPool(OptionalDeviceIndex deviceIndex, MemTypeIndex memTypeIdx);
  void              createDonutPipeline();
  void              scheduleForDeallocation(DeallocationContainer allocation);
  std::optional<uint32_t> getQueueFamilyIndex(vk::QueueFlags flags, std::unordered_set<uint32_t> excludeQueueFamilyIndices);
};
}  // namespace vkdd

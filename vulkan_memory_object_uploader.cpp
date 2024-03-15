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

#include "vulkan_memory_object_uploader.hpp"

#include "command_execution_unit.hpp"
#include "logical_device.hpp"

namespace vkdd {
struct BufferCopy
{
  BufferAllocation        m_srcBufferAllocation;
  vk::Buffer              m_dstBuffer;
  vk::BufferCopy          m_region;
  vk::PipelineStageFlags2 m_dstStageMask;
};

VulkanMemoryObjectUploader::VulkanMemoryObjectUploader(LogicalDevice& logicalDevice)
    : m_logicalDevice(logicalDevice)
{
  vk::SemaphoreTypeCreateInfo timelineSemCreate(vk::SemaphoreType::eTimeline);
  m_syncSem = m_logicalDevice.vkDevice().createSemaphoreUnique({{}, &timelineSemCreate});
}

VulkanMemoryObjectUploader::~VulkanMemoryObjectUploader() {}

void VulkanMemoryObjectUploader::memcpyHost2Buffer(vk::Buffer              dstBuffer,
                                                   size_t                  dstBufferOffset,
                                                   void const*             srcData,
                                                   size_t                  size,
                                                   vk::PipelineStageFlags2 dstStageMask)
{
  std::lock_guard guard(m_mutex);
  vk::BufferCreateInfo stagingBufferCreateInfo({}, size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive);
  BufferAllocation stagingBufferAllocation = m_logicalDevice.allocateStagingBuffer(stagingBufferCreateInfo);
  memcpy(stagingBufferAllocation.m_allocation.mappedMem(), srcData, size);
  m_bufferCopies.emplace_back(
      BufferCopy{std::move(stagingBufferAllocation), dstBuffer, vk::BufferCopy{0, dstBufferOffset, size}, dstStageMask});
}

void VulkanMemoryObjectUploader::prepare(class CommandExecutionUnit& cmdExecUnit)
{
  m_transferCmdBuffer = cmdExecUnit.requestCommandBuffer(m_logicalDevice.getTransferQueueFamilyIndex());
  m_graphicsCmdBuffer = cmdExecUnit.requestCommandBuffer(m_logicalDevice.getGraphicsQueueFamilyIndex());
  cmdExecUnit.pushSignal(m_transferCmdBuffer,
                         {m_syncSem.get(), m_logicalDevice.getCurrentFrameIndex() + 1, vk::PipelineStageFlagBits2::eCopy});
  cmdExecUnit.pushWait(m_graphicsCmdBuffer, {m_syncSem.get(), m_logicalDevice.getCurrentFrameIndex() + 1,
                                             vk::PipelineStageFlagBits2::eAllCommands});
}

void VulkanMemoryObjectUploader::finish()
{
  m_transferCmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  m_graphicsCmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  std::vector<vk::BufferMemoryBarrier2> releases;
  std::vector<vk::BufferMemoryBarrier2> acquisitions;
  for(BufferCopy const& bufferCopy : m_bufferCopies)
  {
    releases.emplace_back(vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eNone,
                          vk::AccessFlagBits2::eNone, m_logicalDevice.getTransferQueueFamilyIndex(),
                          m_logicalDevice.getGraphicsQueueFamilyIndex(), bufferCopy.m_dstBuffer, 0, bufferCopy.m_region.size);
    acquisitions.emplace_back(vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone, bufferCopy.m_dstStageMask,
                              vk::AccessFlagBits2::eMemoryRead, m_logicalDevice.getGraphicsQueueFamilyIndex(),
                              m_logicalDevice.getTransferQueueFamilyIndex(), bufferCopy.m_dstBuffer, 0,
                              bufferCopy.m_region.size);
  }
  for(BufferCopy& bufferCopy : m_bufferCopies)
  {
    m_transferCmdBuffer.copyBuffer(bufferCopy.m_srcBufferAllocation.m_buffer.get(), bufferCopy.m_dstBuffer, bufferCopy.m_region);
    m_logicalDevice.scheduleForDeallocation(std::move(bufferCopy.m_srcBufferAllocation));
    m_transferCmdBuffer.pipelineBarrier2({vk::DependencyFlagBits::eByRegion, {}, releases});
    m_graphicsCmdBuffer.pipelineBarrier2({vk::DependencyFlagBits::eByRegion, {}, acquisitions});
  }
  m_graphicsCmdBuffer.end();
  m_transferCmdBuffer.end();
  m_bufferCopies.clear();
}
}  // namespace vkdd

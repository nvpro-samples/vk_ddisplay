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

#include "triangle_mesh_instance_set.hpp"

#include "logical_device.hpp"
#include "triangle_mesh.hpp"
#include "vulkan_memory_object_uploader.hpp"

namespace vkdd {
TriangleMeshInstanceSet::TriangleMeshInstanceSet(LogicalDevice& logicalDevice, DeviceIndex deviceIndex)
    : m_logicalDevice(logicalDevice)
    , m_deviceIndex(deviceIndex)
    , m_bufferCapacity(0)
{
}

void TriangleMeshInstanceSet::pushInstance(uint32_t uniqueId, Mat4x4f const& model, float shellHeight, float extrusion)
{
  m_instances.emplace_back(DefaultInstance{model, model.invert(), uniqueId, shellHeight, extrusion});
}

void TriangleMeshInstanceSet::endInstanceCollection()
{
  if(m_bufferCapacity < m_instances.size())
  {
    m_logicalDevice.scheduleForDeallocation(std::move(m_bufferAllocation));
    m_bufferCapacity = std::max((uint32_t)m_instances.size(), std::max(16U, 2U * m_bufferCapacity));
    vk::BufferCreateInfo instanceBufferCreateInfo({}, m_bufferCapacity * sizeof(DefaultInstance),
                                                  vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);
    m_bufferAllocation =
        m_logicalDevice.allocateBuffer(m_deviceIndex, instanceBufferCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
  }
}

void TriangleMeshInstanceSet::updateDeviceMemory(vk::CommandBuffer transferCmdBuffer, vk::CommandBuffer graphicsCmdBuffer)
{
  BufferAllocation allocation = m_logicalDevice.allocateStagingBuffer(
      {{}, m_instances.size() * sizeof(DefaultInstance), vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, {}});
  memcpy(allocation.m_allocation.mappedMem(), m_instances.data(), m_instances.size() * sizeof(DefaultInstance));
  vk::BufferCopy copy(0, 0, m_instances.size() * sizeof(DefaultInstance));
  transferCmdBuffer.copyBuffer(allocation.m_buffer.get(), m_bufferAllocation.m_buffer.get(), copy);
  vk::BufferMemoryBarrier2 releaseFromTransferBarrier(
      vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eCopy,
      vk::AccessFlagBits2::eNone, m_logicalDevice.getTransferQueueFamilyIndex(), m_logicalDevice.getGraphicsQueueFamilyIndex(),
      m_bufferAllocation.m_buffer.get(), 0, m_instances.size() * sizeof(DefaultInstance));
  transferCmdBuffer.pipelineBarrier2({vk::DependencyFlagBits::eByRegion, {}, releaseFromTransferBarrier, {}});

  vk::BufferMemoryBarrier2 acquireByGraphicsBarrier(
      vk::PipelineStageFlagBits2::eVertexAttributeInput, vk::AccessFlagBits2::eNone,
      vk::PipelineStageFlagBits2::eVertexAttributeInput, vk::AccessFlagBits2::eMemoryRead,
      m_logicalDevice.getTransferQueueFamilyIndex(), m_logicalDevice.getGraphicsQueueFamilyIndex(),
      m_bufferAllocation.m_buffer.get(), 0, m_instances.size() * sizeof(DefaultInstance));
  graphicsCmdBuffer.pipelineBarrier2({vk::DependencyFlagBits::eByRegion, {}, acquireByGraphicsBarrier, {}});
  m_logicalDevice.scheduleForDeallocation(std::move(allocation));
}

void TriangleMeshInstanceSet::draw(vk::CommandBuffer cmdBuffer, TriangleMesh& triangleMesh)
{
  if(!m_instances.empty())
  {
    cmdBuffer.bindVertexBuffers(0, {triangleMesh.getVertexBuffer(), this->getBuffer()}, {0, this->getBufferOffset()});
    cmdBuffer.bindIndexBuffer(triangleMesh.getIndexBuffer(), 0, vk::IndexType::eUint32);
    cmdBuffer.drawIndexed(triangleMesh.getNumIndices(), (uint32_t)m_instances.size(), 0, 0, 0);
  }
}
}  // namespace vkdd
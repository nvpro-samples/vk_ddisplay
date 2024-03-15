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

namespace vkdd {

struct DefaultInstance
{
  Mat4x4f  m_model;
  Mat4x4f  m_invModel;
  uint32_t m_uniqueId;
  float    m_shellHeight;
  float    m_extrusion;
};

class TriangleMeshInstanceSet
{
public:
  TriangleMeshInstanceSet(class LogicalDevice& logicalDevice, DeviceIndex deviceIndex);

  vk::Buffer     getBuffer() const { return m_bufferAllocation.m_buffer.get(); }
  vk::DeviceSize getBufferOffset() const { return 0; }
  vk::DeviceSize getBufferSize() const { return m_instances.size() * sizeof(DefaultInstance); }
  void           beginInstanceCollection() { m_instances.clear(); }
  void           pushInstance(uint32_t uniqueId, Mat4x4f const& model, float shellHeight, float extrusion);
  void           endInstanceCollection();
  uint32_t       getNumInstances() const { return (uint32_t)m_instances.size(); }
  void           updateDeviceMemory(vk::CommandBuffer transferCmdBuffer, vk::CommandBuffer graphicsCmdBuffer);
  void           draw(vk::CommandBuffer cmdBuffer, class TriangleMesh& triangleMesh);

private:
  LogicalDevice&               m_logicalDevice;
  DeviceIndex                  m_deviceIndex;
  std::vector<DefaultInstance> m_instances;
  BufferAllocation             m_bufferAllocation;
  uint32_t                     m_bufferCapacity;
};
}  // namespace vkdd
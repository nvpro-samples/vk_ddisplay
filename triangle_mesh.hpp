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
#include "vulkan_memory_pool.hpp"

namespace vkdd {
struct DefaultVertex
{
  float m_pos[3];
  float m_normal[3];
  float m_tex[2];
};

class TriangleMesh
{
public:
  struct TimelineSemaphoreTarget
  {
    vk::Semaphore m_semaphore;
    uint64_t      m_targetValue;
  };

  TriangleMesh(class LogicalDevice& logicalDevice, DeviceIndex deviceIndex);

  void       buildTorus(uint32_t numTesselationsX, uint32_t numTesselationsY);
  void       buildSphere(uint32_t numTesselationsX, uint32_t numTesselationsY);
  vk::Buffer getVertexBuffer() const { return m_vertexBuffer.m_buffer.get(); }
  vk::Buffer getIndexBuffer() const { return m_indexBuffer.m_buffer.get(); }
  uint32_t   getNumIndices() const { return m_numIndices; }
  FrameIndex getAvailableFrameIndex() const { return m_availableFrameIndex; }

private:
  LogicalDevice&   m_logicalDevice;
  DeviceIndex      m_deviceIndex;
  BufferAllocation m_vertexBuffer;
  BufferAllocation m_indexBuffer;
  uint32_t         m_numIndices;
  FrameIndex       m_availableFrameIndex;

  void buildParametric(std::function<DefaultVertex(float s, float t)> getVertex, uint32_t numTesselationsS, uint32_t numTesselationsT);
  void buildBuffers(std::vector<uint32_t> const& indices, std::vector<DefaultVertex> const& vertices);
};
}  // namespace vkdd
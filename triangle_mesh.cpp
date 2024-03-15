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

#include "triangle_mesh.hpp"

#include "logical_device.hpp"
#include "vulkan_memory_object_uploader.hpp"

namespace vkdd {
TriangleMesh::TriangleMesh(LogicalDevice& logicalDevice, DeviceIndex deviceIndex)
    : m_logicalDevice(logicalDevice)
    , m_deviceIndex(deviceIndex)
    , m_availableFrameIndex((FrameIndex)-1)
{
}

void TriangleMesh::buildTorus(uint32_t numTesselationsX, uint32_t numTesselationsY)
{
  float r = 0.125f;
  float R = 0.375f;
  return this->buildParametric(
      [&](float s, float t) {
        float sinPhi   = std::sinf(M_2PIf * s);
        float cosPhi   = std::cosf(M_2PIf * s);
        float sinTheta = std::sinf(M_2PIf * t);
        float cosTheta = std::cosf(M_2PIf * t);
        float px       = cosTheta * (R + r * cosPhi);
        float py       = sinTheta * (R + r * cosPhi);
        float pz       = r * sinPhi;
        float tx       = -cosTheta * r * sinPhi;
        float ty       = -sinTheta * r * sinPhi;
        float tz       = r * cosPhi;
        float bx       = -sinTheta * (R + r * cosPhi);
        float by       = cosTheta * (R + r * cosPhi);
        float bz       = 0.0f;
        float nx       = by * tz - bz * ty;
        float ny       = bz * tx - bx * tz;
        float nz       = bx * ty - by * tx;
        return DefaultVertex{{px, py, pz}, {nx, ny, nz}, {s, t}};
      },
      numTesselationsX, numTesselationsY);
}

void TriangleMesh::buildSphere(uint32_t numTesselationsX, uint32_t numTesselationsY)
{
  float r = 1.0f;
  return this->buildParametric(
      [&](float s, float t) {
        float sinPhi   = std::sinf(M_2PIf * s);
        float cosPhi   = std::cosf(M_2PIf * s);
        float sinTheta = std::sinf(M_2PIf * t);
        float cosTheta = std::cosf(M_2PIf * t);
        float px       = r * sinPhi * cosTheta;
        float py       = r * sinPhi * sinTheta;
        float pz       = r * cosPhi;
        return DefaultVertex{{px, py, pz}, {px, py, pz}, {s, t}};
      },
      numTesselationsX, numTesselationsY);
}

void TriangleMesh::buildParametric(std::function<DefaultVertex(float s, float t)> getVertex, uint32_t numTesselationsS, uint32_t numTesselationsT)
{
  std::vector<DefaultVertex> vertices;
  for(uint32_t it = 0; it < numTesselationsT; ++it)
  {
    float t = (float)it / (float)(numTesselationsT - 1);
    for(uint32_t is = 0; is < numTesselationsS; ++is)
    {
      float s = (float)is / (float)(numTesselationsS - 1);
      vertices.emplace_back(getVertex(s, t));
    }
  }
  std::vector<uint32_t> indices;
  for(uint32_t i = 0; i < numTesselationsT - 1; ++i)
  {
    for(uint32_t j = 0; j < numTesselationsS; ++j)
    {
      indices.emplace_back(i * numTesselationsS + j);
      indices.emplace_back((i + 1) * numTesselationsS + j);
    }
    indices.emplace_back(0xffffffff);
  }
  indices.pop_back();
  this->buildBuffers(indices, vertices);
}

void TriangleMesh::buildBuffers(std::vector<uint32_t> const& indices, std::vector<DefaultVertex> const& vertices)
{
  m_numIndices = (uint32_t)indices.size();
  vk::BufferCreateInfo indexBufferCreateInfo({}, indices.size() * sizeof(uint32_t),
                                             vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                             vk::SharingMode::eExclusive, {});
  m_indexBuffer = m_logicalDevice.allocateBuffer(m_deviceIndex, indexBufferCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
  m_logicalDevice.getUploader().memcpyHost2Buffer(m_indexBuffer.m_buffer.get(), 0, indices.data(),
                                                  indices.size() * sizeof(uint32_t), vk::PipelineStageFlagBits2::eIndexInput);

  vk::BufferCreateInfo vertexBufferCreateInfo({}, vertices.size() * sizeof(DefaultVertex),
                                              vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                              vk::SharingMode::eExclusive, {});
  m_vertexBuffer = m_logicalDevice.allocateBuffer(m_deviceIndex, vertexBufferCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
  m_logicalDevice.getUploader().memcpyHost2Buffer(m_vertexBuffer.m_buffer.get(), 0, vertices.data(),
                                                  vertices.size() * sizeof(DefaultVertex),
                                                  vk::PipelineStageFlagBits2::eVertexAttributeInput);
  m_availableFrameIndex = m_logicalDevice.getCurrentFrameIndex() + 1;
}
}  // namespace vkdd
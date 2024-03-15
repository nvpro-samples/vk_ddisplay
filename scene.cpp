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

#include "scene.hpp"

namespace vkdd {
Scene::Node::Node(NodeType nodeType, Vec3f scaling, Angle roll, Angle pitch, Angle yaw, Vec3f translation)
    : m_nodeType(nodeType)
    , m_scaling(scaling)
    , m_roll(roll)
    , m_pitch(pitch)
    , m_yaw(yaw)
    , m_translation(translation)
{
  static uint32_t nextId = 0;
  m_id                   = nextId++;
}

void Scene::Node::setRotation(Angle roll, Angle pitch, Angle yaw)
{
  m_roll  = roll;
  m_pitch = pitch;
  m_yaw   = yaw;
}

Mat4x4f Scene::Node::createModel() const
{
  return Mat4x4f::affineLinearTransformation(m_scaling, m_roll, m_pitch, m_yaw, m_translation);
}

Scene::Scene()
    : m_desiredNumDonutsX(15)
    , m_desiredNumDonutsY(9)
    , m_numDonutsX(0)
    , m_numDonutsY(0)
{
  m_camera.m_pos  = {0.0f, 0.0f, -4.0f};
  m_camera.m_view = Mat4x4f::translation(m_camera.m_pos).invert();
  this->setPerspectiveCamera(16.0f / 9.0f, Angle::degree(90.0f), 1e-2f, 1e2f);
}

void Scene::setPerspectiveCamera(float aspect, Angle fov, float nearZ, float farZ)
{
  m_camera.m_aspect = aspect;
  m_camera.m_fov    = fov;
  m_camera.m_nearZ  = nearZ;
  m_camera.m_farZ   = farZ;
  m_camera.m_proj = Mat4x4f::perspectiveProjection(m_camera.m_fov, m_camera.m_aspect, m_camera.m_nearZ, m_camera.m_farZ);
  this->rebuild();
}

void Scene::update(float millis)
{
  m_runtimeMillis += millis;
  ++m_numUpdates;
  if(m_numDonutsX != m_desiredNumDonutsX || m_numDonutsY != m_desiredNumDonutsY)
  {
    this->rebuild();
  }
  for(uint32_t i = 0; i < m_geometryNodes.size(); ++i)
  {
    srand(i);
    float r0 = 1e-2f * (float)(20 + rand() % 80);
    float r1 = 1e-2f * (float)(20 + rand() % 80);
    float r2 = 1e-2f * (float)(20 + rand() % 80);
    m_geometryNodes[i].setRotation(Angle::radians(r1 + m_runtimeMillis * 1e-3f * r0),
                                   Angle::radians(r2 + m_runtimeMillis * 1e-3f * r1),
                                   Angle::radians(r0 + m_runtimeMillis * 1e-3f * r2));
  }
}

void Scene::collectVisibleNodes(vk::Viewport globalViewport, vk::Viewport localViewport, std::function<void(Node const& node)> onVisible) const
{
  for(Node const& node : m_geometryNodes)
  {
    onVisible(node);
  }
}

void Scene::fillDonutPlane(float z, uint32_t numDonutsX, uint32_t numDonutsY)
{
  float backPlaneX            = 2.0f * std::fabsf(z - m_camera.m_pos.z) * std::tanf(0.5f * m_camera.m_fov.radians());
  float backPlaneY            = backPlaneX / m_camera.m_aspect;
  Vec3f backPlaneTorusScaling = 0.9f * std::min(backPlaneX / (float)numDonutsX, backPlaneY / (float)numDonutsY);
  Vec3f backPlaneTorusSpacing = {backPlaneX / (float)numDonutsX, backPlaneY / (float)numDonutsY, 1.0f};

  for(int32_t y = 0; y < (int32_t)numDonutsY; ++y)
  {
    for(int32_t x = 0; x < (int32_t)numDonutsX; ++x)
    {
      m_geometryNodes.emplace_back(
          NodeType::TORUS, backPlaneTorusScaling, Angle{}, Angle{}, Angle{},
          backPlaneTorusSpacing * Vec3f{(float)x - 0.5f * (float)(numDonutsX - 1), (float)y - 0.5f * (float)(numDonutsY - 1), z});
    }
  }
}

void Scene::rebuild()
{
  m_desiredNumDonutsX = std::max(1, m_desiredNumDonutsX);
  m_desiredNumDonutsY = std::max(1, m_desiredNumDonutsY);
  if(m_desiredNumDonutsX != m_numDonutsX || m_desiredNumDonutsY != m_numDonutsY)
  {
    m_numDonutsX = m_desiredNumDonutsX;
    m_numDonutsY = m_desiredNumDonutsY;
    m_geometryNodes.clear();
    this->fillDonutPlane(0.0f, m_numDonutsX, m_numDonutsY);
    this->fillDonutPlane(-2.0f, 2 * std::max(m_numDonutsX / 4, 1) - 1, 2 * std::max(m_numDonutsY / 4, 1) - 1);
  }
}

void Scene::increaseNumDonutsX()
{
  ++m_numDonutsX;
  this->rebuild();
}

void Scene::decreaseNumDonutsX()
{
  m_numDonutsX = std::max(1, m_numDonutsX - 1);
  this->rebuild();
}

void Scene::increaseNumDonutsY()
{
  ++m_numDonutsY;
  this->rebuild();
}

void Scene::decreaseNumDonutsY()
{
  m_numDonutsY = std::max(1, m_numDonutsY - 1);
  this->rebuild();
}
}  // namespace vkdd
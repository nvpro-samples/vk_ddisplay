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
#include <array>
#include <cmath>
#include <cstring>
#include <stdint.h>

#define M_2PIf (2.0f * 3.141592653589793f)
#define M_PIf 3.1415926535f
#define DEG2RAD(_deg) (M_PIf * (_deg) / 180.0f)

namespace vkdd {
class Angle
{
public:
  static Angle radians(float radians) { return Angle(radians); }
  static Angle degree(float degree) { return Angle(M_PIf * degree / 180.0f); }

  Angle()
      : m_radians(0.0f)
  {
  }

  float         radians() const { return m_radians; }
  float         degree() const { return 180.0f * m_radians / M_PIf; }
  Angle         operator*(float factor) const { return Angle(factor * m_radians); }
  inline Angle& operator*=(float factor);

private:
  Angle(float radians)
      : m_radians(radians)
  {
  }

  float m_radians;
};

Angle& Angle::operator*=(float factor)
{
  m_radians *= factor;
  return *this;
}

static Angle operator*(float factor, Angle angle)
{
  return angle * factor;
}

static float tanf(Angle angle)
{
  return std::tanf(angle.radians());
}

static float sinf(Angle angle)
{
  return std::sinf(angle.radians());
}

static float cosf(Angle angle)
{
  return std::cosf(angle.radians());
}

class Vec3f
{
public:
  float x, y, z;

  Vec3f(float x, float y, float z)
      : x(x)
      , y(y)
      , z(z)
  {
  }

  Vec3f(float singleValue = 0.0f)
      : Vec3f(singleValue, singleValue, singleValue)
  {
  }

  Vec3f  operator+(Vec3f const& right) const { return {x + right.x, y + right.y, z + right.z}; }
  Vec3f  operator-(Vec3f const& right) const { return {x - right.x, y - right.y, z - right.z}; }
  Vec3f  operator*(Vec3f const& factor) const { return {x * factor.x, y * factor.y, z * factor.z}; }
  Vec3f& operator*=(Vec3f const& factor) { return *this = *this * factor; }
  Vec3f  operator*(float factor) const { return {x * factor, y * factor, z * factor}; }
  Vec3f& operator*=(float factor) { return *this = *this * factor; }
};

inline Vec3f operator*(float left, Vec3f const& right)
{
  return {left * right.x, left * right.y, left * right.z};
}

class Vec4f
{
public:
  float x, y, z, w;

  Vec4f(float x, float y, float z, float w)
      : x(x)
      , y(y)
      , z(z)
      , w(w)
  {
  }

  Vec4f(float singleValue = 0.0f)
      : Vec4f(singleValue, singleValue, singleValue, singleValue)
  {
  }

  Vec4f(Vec3f xyz, float w)
      : Vec4f(xyz.x, xyz.y, xyz.z, w)
  {
  }

  Vec4f operator+(Vec4f const& right) const { return {x + right.x, y + right.y, z + right.z, w + right.w}; }
  Vec4f operator-(Vec4f const& right) const { return {x - right.x, y - right.y, z - right.z, w - right.w}; }
};

class Mat4x4f
{
public:
  static Mat4x4f fromRows(Vec4f const& p_row0, Vec4f const& p_row1, Vec4f const& p_row2, Vec4f const& p_row3)
  {
    return {{p_row0.x, p_row1.x, p_row2.x, p_row3.x, p_row0.y, p_row1.y, p_row2.y, p_row3.y, p_row0.z, p_row1.z,
             p_row2.z, p_row3.z, p_row0.w, p_row1.w, p_row2.w, p_row3.w}};
  }
  static Mat4x4f fromColumns(Vec4f const& p_col0, Vec4f const& p_col1, Vec4f const& p_col2, Vec4f const& p_col3)
  {
    return {{p_col0.x, p_col0.y, p_col0.z, p_col0.w, p_col1.x, p_col1.y, p_col1.z, p_col1.w, p_col2.x, p_col2.y,
             p_col2.z, p_col2.w, p_col3.x, p_col3.y, p_col3.z, p_col3.w}};
  }

  static Mat4x4f diagonal(Vec4f const& diagonalValues)
  {
    return Mat4x4f::fromRows({diagonalValues.x, 0.0f, 0.0f, 0.0f}, {0.0f, diagonalValues.y, 0.0f, 0.0f},
                             {0.0f, 0.0f, diagonalValues.z, 0.0f}, {0.0f, 0.0f, 0.0f, diagonalValues.w});
  }

  static Mat4x4f identity() { return Mat4x4f::diagonal(1.0f); }

  static Mat4x4f rotationX(Angle amount)
  {
    float sina = sinf(amount);
    float cosa = cosf(amount);
    return Mat4x4f::fromRows({1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, cosa, -sina, 0.0f}, {0.0f, sina, cosa, 0.0f},
                             {0.0f, 0.0f, 0.0f, 1.0f});
  }

  static Mat4x4f rotationY(Angle amount)
  {
    float sina = sinf(amount);
    float cosa = cosf(amount);
    return Mat4x4f::fromRows({cosa, 0.0f, sina, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {-sina, 0.0f, cosa, 0.0f},
                             {0.0f, 0.0f, 0.0f, 1.0f});
  }

  static Mat4x4f rotationZ(Angle amount)
  {
    float sina = sinf(amount);
    float cosa = cosf(amount);
    return Mat4x4f::fromRows({cosa, -sina, 0.0f, 0.0f}, {sina, cosa, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f},
                             {0.0f, 0.0f, 0.0f, 1.0f});
  }

  static Mat4x4f scaling(Vec3f const& scaling)
  {
    return Mat4x4f::fromRows({scaling.x, 0.0f, 0.0f, 0.0f}, {0.0f, scaling.y, 0.0f, 0.0f},
                             {0.0f, 0.0f, scaling.z, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f});
  }

  static Mat4x4f translation(Vec3f const& translation)
  {
    return Mat4x4f::fromRows({1.0f, 0.0f, 0.0f, translation.x}, {0.0f, 1.0f, 0.0f, translation.y},
                             {0.0f, 0.0f, 1.0f, translation.z}, {0.0f, 0.0f, 0.0f, 1.0f});
  }

  static Mat4x4f affineLinearTransformation(Vec3f const& scaling, Angle roll, Angle pitch, Angle yaw, Vec3f const& translation)
  {
    return Mat4x4f::translation(translation) * Mat4x4f::rotationY(yaw) * Mat4x4f::rotationX(pitch)
           * Mat4x4f::rotationZ(roll) * Mat4x4f::scaling(scaling);
  }

  static Mat4x4f perspectiveProjection(Angle horFov, float aspect, float nearZ, float farZ)
  {
    float tana = tanf(0.5f * horFov);
    float tanb = tana / aspect;
    return Mat4x4f::fromRows({1.0f / tana, 0.0f, 0.0f, 0.0f}, {0.0f, -1.0f / tanb, 0.0f, 0.0f},
                             {0.0f, 0.0f, farZ / (farZ - nearZ), -nearZ * farZ / (farZ - nearZ)}, {0.0f, 0.0f, 1.0f, 0.0f});
  }

  float m_values[16];

  Mat4x4f() { memset(m_values, 0, 16 * sizeof(float)); }
  Mat4x4f(std::array<float, 16> const& values) { memcpy(m_values, values.data(), 16 * sizeof(float)); }

  inline void    set(uint32_t row, uint32_t col, float value) { m_values[4 * col + row] = value; }
  inline float   get(uint32_t row, uint32_t col) const { return m_values[4 * col + row]; }
  inline Mat4x4f operator*(Mat4x4f const& right) const;
  inline Vec4f   operator*(Vec4f const& right) const;
  inline Vec3f   transformCoord(Vec3f const& v) const;
  inline Vec3f   transformVector(Vec3f const& v) const;
  inline float   determinant() const;
  inline Mat4x4f invert(float* optOutDeterminant = nullptr) const;

private:
  inline float determinant(uint32_t ignoreRow, uint32_t ignoreCol) const;
};

Mat4x4f Mat4x4f::operator*(Mat4x4f const& right) const
{
  Mat4x4f result;
  for(uint32_t r = 0; r < 4; ++r)
  {
    for(uint32_t c = 0; c < 4; ++c)
    {
      float v = 0.0f;
      for(uint32_t i = 0; i < 4; ++i)
      {
        v += this->get(r, i) * right.get(i, c);
      }
      result.set(r, c, v);
    }
  }
  return result;
}

Vec4f Mat4x4f::operator*(Vec4f const& right) const
{
  return {this->get(0, 0) * right.x + this->get(0, 1) * right.y + this->get(0, 2) * right.z + this->get(0, 3) * right.w,
          this->get(1, 0) * right.x + this->get(1, 1) * right.y + this->get(1, 2) * right.z + this->get(1, 3) * right.w,
          this->get(2, 0) * right.x + this->get(2, 1) * right.y + this->get(2, 2) * right.z + this->get(2, 3) * right.w,
          this->get(3, 0) * right.x + this->get(3, 1) * right.y + this->get(3, 2) * right.z + this->get(3, 3) * right.w};
}

Vec3f Mat4x4f::transformCoord(Vec3f const& v) const
{
  Vec4f r = *this * Vec4f(v, 1.0f);
  return {r.x / r.w, r.y / r.w, r.z / r.w};
}

Vec3f Mat4x4f::transformVector(Vec3f const& v) const
{
  return {this->get(0, 0) * v.x + this->get(0, 1) * v.y + this->get(0, 2) * v.z,
          this->get(1, 0) * v.x + this->get(1, 1) * v.y + this->get(1, 2) * v.z,
          this->get(2, 0) * v.x + this->get(2, 1) * v.y + this->get(2, 2) * v.z};
}

float Mat4x4f::determinant() const
{
  return this->get(0, 0) * this->determinant(0, 0) - this->get(0, 1) * this->determinant(0, 1)
         + this->get(0, 2) * this->determinant(0, 2) - this->get(0, 3) * this->determinant(0, 3);
}

float Mat4x4f::determinant(uint32_t ignoreRow, uint32_t ignoreCol) const
{
  int32_t rows[3] = {ignoreRow < 1 ? 1 : 0, ignoreRow < 2 ? 2 : 1, ignoreRow < 3 ? 3 : 2};
  int32_t cols[3] = {ignoreCol < 1 ? 1 : 0, ignoreCol < 2 ? 2 : 1, ignoreCol < 3 ? 3 : 2};
  return this->get(rows[0], cols[0]) * this->get(rows[1], cols[1]) * this->get(rows[2], cols[2])
         + this->get(rows[0], cols[1]) * this->get(rows[1], cols[2]) * this->get(rows[2], cols[0])
         + this->get(rows[0], cols[2]) * this->get(rows[1], cols[0]) * this->get(rows[2], cols[1])
         - this->get(rows[0], cols[2]) * this->get(rows[1], cols[1]) * this->get(rows[2], cols[0])
         - this->get(rows[0], cols[0]) * this->get(rows[1], cols[2]) * this->get(rows[2], cols[1])
         - this->get(rows[0], cols[1]) * this->get(rows[1], cols[0]) * this->get(rows[2], cols[2]);
}

Mat4x4f Mat4x4f::invert(float* optOutDeterminant) const
{
  float   det = this->determinant();
  Mat4x4f inverse;
  for(uint32_t r = 0; r < 4; ++r)
  {
    for(uint32_t c = 0; c < 4; ++c)
    {
      inverse.set(c, r, ((r + c) % 2 == 0 ? 1.0f : -1.0f) / det * this->determinant(r, c));
    }
  }
  if(optOutDeterminant)
  {
    *optOutDeterminant = det;
  }
  return inverse;
}

class Aabb
{
public:
  Vec3f const& getMin() const { return m_min; }
  Vec3f const& getMax() const { return m_min; }
  Vec3f        getSize() const { return m_max - m_min; }

private:
  Vec3f m_min;
  Vec3f m_max;
};

template <typename T>
T lerp(T const& a, T const& b, float t)
{
  return a * (1.0f - t) + t * b;
}
}  // namespace vkdd

#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <cmath>

struct rgb
{
	float Red, Green, Blue;
	__host__ __device__ rgb(float r, float g, float b) : Red(r), Green(g), Blue(b) {}
};

class vec3
{
public:
	__host__ __device__ vec3() : e{ 0.0f, 0.0f, 0.0f } {}
	__host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}
	__host__ __device__ vec3(const rgb& color) : e{ color.Red / 255.0f, color.Green / 255.0f, color.Blue / 255.0f } {}

	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }
	__host__ __device__ inline float z() const { return e[2]; }

	__host__ __device__ inline float r() const { return e[0]; }
	__host__ __device__ inline float g() const { return e[1]; }
	__host__ __device__ inline float b() const { return e[2]; }

	__host__ __device__ inline const vec3& operator+() const { return *this; }
	__host__ __device__ inline vec3   operator-()      const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ inline float  operator[](int32_t i) const { return e[i]; }
	__host__ __device__ inline float& operator[](int32_t i)       { return e[i]; }

	__host__ __device__ inline vec3& operator+=(const vec3& other);
	__host__ __device__ inline vec3& operator-=(const vec3& other);
	__host__ __device__ inline vec3& operator*=(const vec3& other);
	__host__ __device__ inline vec3& operator/=(const vec3& other);

	__host__ __device__ inline vec3& operator*=(const float val);
	__host__ __device__ inline vec3& operator/=(const float val);

	__host__ __device__ inline float Length()        const { return std::sqrt(LengthSquared()); }
	__host__ __device__ inline float LengthSquared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
	__host__ __device__ inline void  MakeUnitVector();

public:
	float e[3];
};

// vec3 utility functions
inline std::ostream& operator<<(std::ostream& out, const vec3& v)
{
	return out << v.e[0] << " " << v.e[1] << " " << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v)
{
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v)
{
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v)
{
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& u, const vec3& v) 
{
	return vec3(u.e[0] / v.e[0], u.e[1] / v.e[1], u.e[2] / v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v)
{
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t)
{
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
	return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline float Dot(const vec3& u, const vec3& v)
{
	return u.e[0] * v.e[0]
		 + u.e[1] * v.e[1]
		 + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 Cross(const vec3& u, const vec3& v)
{
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
				u.e[2] * v.e[0] - u.e[0] * v.e[2],
				u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 UnitVector(const vec3& v)
{
	return v / v.Length();
}

// member functions
__host__ __device__ inline vec3& vec3::operator += (const vec3& other)
{
	e[0] += other.e[0];
	e[1] += other.e[1];
	e[2] += other.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator -= (const vec3& other)
{
	e[0] -= other.e[0];
	e[1] -= other.e[1];
	e[2] -= other.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator *= (const vec3& other)
{
	e[0] *= other.e[0];
	e[1] *= other.e[1];
	e[2] *= other.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator /= (const vec3& other)
{
	e[0] /= other.e[0];
	e[1] /= other.e[1];
	e[2] /= other.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator *= (const float val)
{
	e[0] *= val;
	e[1] *= val;
	e[2] *= val;
	return *this;
}

__host__ __device__ inline vec3& vec3::operator /= (const float val)
{
	e[0] /= val;
	e[1] /= val;
	e[2] /= val;
	return *this;
}

__host__ __device__ inline void vec3::MakeUnitVector()
{
	float v = 1.0f / Length();
	e[0] *= v;
	e[1] *= v;
	e[2] *= v;
}

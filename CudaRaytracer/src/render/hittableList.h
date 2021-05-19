#pragma once

#include "hittable.h"

class HittableList : public Hittable
{
public:
	__device__ HittableList() : m_List(nullptr), m_ListSize(0) {}
	__device__ HittableList(Hittable** list, int32_t listSize) : m_List(list), m_ListSize(listSize) {}

	__device__ bool Hit(const Ray& ray, float tMin, float tMax, HitRecords& rec) const override
	{
		HitRecords tempRec;
		bool  isHit = false;
		float closest = tMax;

		for (int i = 0; i < m_ListSize; i++)
		{
			// Find the closest point that was hit
			if (m_List[i]->Hit(ray, tMin, closest, tempRec))
			{
				isHit = true;
				closest = tempRec.t;
				rec = tempRec;
			}
		}

		return isHit;
	}

public:
	Hittable** m_List;
	int32_t m_ListSize;
};

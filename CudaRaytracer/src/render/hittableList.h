#include "hittable.h"


class HittableList : public Hittable
{
public:
	__device__ HittableList() = default;
	__device__ HittableList(Hittable** list, int32_t listSize)
	{
		m_List     = list;
		m_ListSize = listSize;
	}

	__device__ bool Hit(const Ray& ray, float tMin, float tMax, HitRecords& rec) const override;

public:
	Hittable** m_List;
	int32_t m_ListSize;
};

__device__ bool HittableList::Hit(const Ray& ray, float tMin, float tMax, HitRecords& rec) const
{
	HitRecords tempRec;
	bool  isHit = false;
	float closest = tMax;

	for (int i = 0; i < m_ListSize; i++)
	{
		if (m_List[i]->Hit(ray, tMin, closest, tempRec))
		{
			isHit = true;
			closest = tempRec.t;
			rec = tempRec;
		}
	}
	return isHit;
}

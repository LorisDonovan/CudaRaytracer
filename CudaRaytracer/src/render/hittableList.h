#include "hittable.h"


class HittableList : public Hittable
{
public:
	__device__ HittableList() = default;
	__device__ HittableList(Hittable** list, int32_t listSize);

	__device__ bool Hit(const Ray& ray, float tMin, float tMax, HitRecords& rec) const override;

public:
	Hittable** m_List;
	int32_t m_ListSize;
};


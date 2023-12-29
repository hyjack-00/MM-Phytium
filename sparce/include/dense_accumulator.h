#ifndef DENSE_ACCUMULATOR_H
#include <arm_neon.h>
#include "popcnt.h"

template<typename CSROrdinal, typename Value, typename DHMValue, typename LOrdType>
class DenseHashMap {
    public:
        DHMValue * densearray;
        LOrdType * keys;

        CSROrdinal densearray_size;
        CSROrdinal keys_size;

        CSROrdinal used_densearray_size;

        CSROrdinal count;
        DHMValue initial_value;

        DenseHashMap(CSROrdinal width, CSROrdinal max_insertion, DHMValue _initial_value = 0)
            : densearray(NULL), keys(NULL), densearray_size(width), initial_value(_initial_value) {
            keys_size = std::min(width, max_insertion);

            densearray = new DHMValue[densearray_size];
            keys = new LOrdType[keys_size];

            for (CSROrdinal i = 0; i < densearray_size; ++i)
                densearray[i] = initial_value;

            used_densearray_size = 0;
            count = 0;
        }

        ~DenseHashMap() {
            delete [] densearray;
            delete [] keys;
        }

        DenseHashMap(const DenseHashMap&) = delete;
        DenseHashMap(DenseHashMap&&) = delete;
        DenseHashMap& operator=(const DenseHashMap&) = delete;
        DenseHashMap& operator=(DenseHashMap&&) = delete;

        void insert(LOrdType key) {
            if (densearray[key] == initial_value) {
                keys[used_densearray_size++] = key;
                densearray[key] = 1;
            }
        }

        void insertOr(LOrdType key, DHMValue val) {
            assert(val != initial_value);
            assert(key < densearray_size);
            if (densearray[key] == initial_value) {
                assert(used_densearray_size < keys_size);
                keys[used_densearray_size++] = key;
                densearray[key] = val;
            } else
                densearray[key] |= val;
        }

        void insertInc(LOrdType key, Value val) {
            assert(val != initial_value);
            if (densearray[key] == initial_value) {
                keys[used_densearray_size++] = key;
                densearray[key] = val;
            } else
                densearray[key] += val;
        }

        CSROrdinal getUsedSize() {
            return used_densearray_size;
        }

        CSROrdinal getUsedSizeOr(bool reset=true) {
            CSROrdinal size = 0;
            for ( CSROrdinal i=0; i<used_densearray_size; ++i ) {
                size += bitcounts(densearray[ keys[i] ]);
                if (reset)
                    densearray[ keys[i] ] = initial_value;
            }
            return size;
        }

        bool getKeyValue(LOrdType & key, Value & val) {
            if (count >= used_densearray_size) return false;
            key = keys[count];
            val = densearray[key];
            densearray[key] = initial_value;
            count++;
            return true;
        }

        void resetSize() {
            used_densearray_size = 0;
            count = 0;
        }

        void reset() {
            for ( CSROrdinal i=0; i<used_densearray_size; ++i)
                densearray[ keys[i] ] = initial_value;
            resetSize();
        }
};


template<>
class DenseHashMap<uint32_t, float32_t, float32_t, uint16_t> {
public:
    float32_t* densearray;
    uint16_t* keys;

    uint32_t densearray_size;
    uint32_t keys_size;

    uint32_t used_densearray_size;

    uint32_t count;
    float32_t initial_value;

    DenseHashMap(uint32_t width, uint32_t max_insertion, float32_t _initial_value = 0)
        : densearray(NULL), keys(NULL), densearray_size(width), initial_value(_initial_value) {
        keys_size = std::min(width, max_insertion);

        densearray = new float32_t[densearray_size];
        keys = new uint16_t[keys_size];

        for (uint32_t i = 0; i < densearray_size; ++i)
            densearray[i] = initial_value;

        used_densearray_size = 0;
        count = 0;
    }

    ~DenseHashMap() {
        delete[] densearray;
        delete[] keys;
    }

    DenseHashMap(const DenseHashMap&) = delete;
    DenseHashMap(DenseHashMap&&) = delete;
    DenseHashMap& operator=(const DenseHashMap&) = delete;
    DenseHashMap& operator=(DenseHashMap&&) = delete;

    void insertInc(uint16_t key, float32_t val) {
        assert(val != initial_value);
        if (densearray[key] == initial_value) {
            keys[used_densearray_size++] = key;
            densearray[key] = val;
        }
        else
            densearray[key] += val;
    }

    uint32_t getUsedSize() {
        return used_densearray_size;
    }

    bool getKeyValuex4(uint32x4_t& key, float32x4_t& val) {
        if (count + 3 >= used_densearray_size) return false;
        key = vmovl_u16(vld1_u16(keys + count));
        vld1q_lane_f32(densearray + vgetq_lane_u32(key, 0), val, 0);
        vld1q_lane_f32(densearray + vgetq_lane_u32(key, 1), val, 1);
        vld1q_lane_f32(densearray + vgetq_lane_u32(key, 2), val, 2);
        vld1q_lane_f32(densearray + vgetq_lane_u32(key, 3), val, 3);

        densearray[vgetq_lane_u32(key, 0)] = initial_value;
        densearray[vgetq_lane_u32(key, 1)] = initial_value;
        densearray[vgetq_lane_u32(key, 2)] = initial_value;
        densearray[vgetq_lane_u32(key, 3)] = initial_value;

        count += 4;
        return true;
    }

    bool getKeyValue(uint16_t& key, float32_t& val) {
        if (count >= used_densearray_size) return false;
        key = keys[count];
        val = densearray[key];
        densearray[key] = initial_value;
        count++;
        return true;
    }

    void resetSize() {
        used_densearray_size = 0;
        count = 0;
    }

    void reset() {
        for (uint32_t i = 0; i < used_densearray_size; ++i)
            densearray[keys[i]] = initial_value;
        resetSize();
    }
};

#endif

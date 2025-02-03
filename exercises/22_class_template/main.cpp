#include "../exercise.h"
#include <cstring>
// READ: 类模板 <https://zh.cppreference.com/w/cpp/language/class_template>

template<class T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        for (int i = 0; i < 4; i++) {
            shape[i] = shape_[i];
            size *= shape[i];
        }
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));
    }

    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    Tensor4D &operator+=(Tensor4D const &others) {
        // 检查广播兼容性
        for (int i = 0; i < 4; i++) {
            if (others.shape[i] != 1 && others.shape[i] != shape[i]) {
                throw std::invalid_argument("Shape mismatch for broadcasting");
            }
        }

        // 计算 this 张量各个维度的“步幅”（即每前进一维所对应的一维数组中连续元素个数）
        unsigned int stride0 = shape[1] * shape[2] * shape[3];
        unsigned int stride1 = shape[2] * shape[3];
        unsigned int stride2 = shape[3];
        unsigned int stride3 = 1;

        // 对于 others 张量也同样计算步幅（注意：即使某一维的长度为 1，步幅依然按照原始 shape 计算）
        unsigned int stride0_o = others.shape[1] * others.shape[2] * others.shape[3];
        unsigned int stride1_o = others.shape[2] * others.shape[3];
        unsigned int stride2_o = others.shape[3];
        unsigned int stride3_o = 1;

        // 采用 4 重循环遍历 this 的每一个元素
        for (unsigned int i = 0; i < shape[0]; ++i) {
            for (unsigned int j = 0; j < shape[1]; ++j) {
                for (unsigned int k = 0; k < shape[2]; ++k) {
                    for (unsigned int l = 0; l < shape[3]; ++l) {
                        // 计算 this 张量中当前元素的线性索引
                        unsigned int idx = i * stride0 + j * stride1 + k * stride2 + l * stride3;

                        // 对于 others 张量的索引，判断每一维是否需要广播：
                        // 如果 others 在某一维的长度为 1，则该维度的索引永远为 0，
                        // 否则使用 this 张量中的对应索引。
                        unsigned int i_o = (others.shape[0] == 1) ? 0 : i;
                        unsigned int j_o = (others.shape[1] == 1) ? 0 : j;
                        unsigned int k_o = (others.shape[2] == 1) ? 0 : k;
                        unsigned int l_o = (others.shape[3] == 1) ? 0 : l;

                        // 计算 others 张量中对应元素的线性索引
                        unsigned int idx_o = i_o * stride0_o + j_o * stride1_o + k_o * stride2_o + l_o * stride3_o;

                        // 完成加法
                        data[idx] += others.data[idx_o];
                    }
                }
            }
        }
        return *this;
    }
};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D(shape, data);
        auto t1 = Tensor4D(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}

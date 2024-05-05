#include <cmath>
#include <iostream>
#include <chrono>
#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <vector>

void _mm256_print_epi8(__m256i vec)
{
    char temp[32];
    _mm256_storeu_si256((__m256i*)&temp[0], vec);
    int i;
    for(i=0; i<32; i++)
        printf(" %f,", temp[i]);
}

void _mm512_print_epi32(__m512i vec) {
    char temp[64];
    _mm512_store_si512((__m512i*)&temp[0], vec);
    int i;
    int32_t* p = (int32_t*)temp;
    for (i = 0; i < 16; i++)
        printf("%d,", p[i]);
    printf("\n");
}

void _mm128_print_epi32(__m128i vec) {
    char temp[16];
    _mm_store_si128((__m128i*)&temp[0], vec);
    int i;
    int32_t* p = (int32_t*)temp;
    for (i = 0; i < 4; i++)
        printf("%d,", p[i]);
    printf("\n");
}

inline int 
SquareSum(const int8_t* x, int d) {
    d = 32;
    __m512i sum = _mm512_setzero_si512();
    for (int i = 0; i < d; i += 32) {
        __m256i xx = _mm256_loadu_si256((__m256i *)(x + i));
        __m512i xx_ext = _mm512_cvtepi8_epi16(xx);
        sum = _mm512_add_epi32(_mm512_madd_epi16(xx_ext, xx_ext), sum);
    }
    // reduce for 16 int32 integers in AVX512
    __m256i sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(sum, 0), _mm512_extracti32x8_epi32(sum, 1));
    __m128i sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh), _mm256_extracti128_si256(sumh, 1));
    __m128i tmp = _mm_hadd_epi32(sumhh, sumhh);
    _mm128_print_epi32(tmp);
    return _mm_extract_epi32(tmp, 0) + _mm_extract_epi32(tmp, 1);
}

inline int 
IPSQ8(const int8_t* x, const int8_t *y, int d) {
    d = 32;
    __m512i sum = _mm512_setzero_si512();
    for (int i = 0; i < d; i += 32) {
        __m256i xx = _mm256_loadu_si256((__m256i *)(x + i));
        __m256i yy = _mm256_loadu_si256((__m256i *)(y + i));
        __m512i xx_ext = _mm512_cvtepi8_epi16(xx);
        __m512i yy_ext = _mm512_cvtepi8_epi16(yy);
        sum = _mm512_add_epi32(_mm512_madd_epi16(xx_ext, yy_ext), sum);   
    }
    // reduce for 16 int32 integers in AVX512
    __m256i sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(sum, 0), _mm512_extracti32x8_epi32(sum, 1));
    __m128i sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh), _mm256_extracti128_si256(sumh, 1));
    __m128i tmp = _mm_hadd_epi32(sumhh, sumhh);
    _mm128_print_epi32(tmp);
    return _mm_extract_epi32(tmp, 0) + _mm_extract_epi32(tmp, 1);
}

inline int NormalIP(const int8_t* x, const int8_t* y, int d) {
    d = 32;
    int32_t res = 0;
    for (int i = 0; i < d; i++) {
        int32_t tmp = (int32_t)x[i] * (int32_t)y[i];
        res += tmp;
    }
    printf("%d\n", res);
    return res;
}

inline int NormalSquareSum(const int8_t* x, int d) {
    d = 32;
    int32_t res = 0;
    for (int i = 0; i < d; i++) {
        int32_t tmp = (int32_t)x[i];
        res += tmp * tmp;
    }
    printf("%d\n", res);
    return res;
}

inline float
L2SqrSQ8_sym(const int8_t *x, const int8_t *y, int d) {
    size_t i;
    d = 32;
    // __m512i sum = _mm512_setzero_si512();
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < d; i += 16) {
      // convert to 512bit
        __m128i xx = _mm_loadu_si128((__m128i *)(x + i));
        __m128i yy = _mm_loadu_si128((__m128i *)(y + i));
        // __m512i xx_ext = _mm512_cvtepi8_epi32(xx);
        // __m512i yy_ext = _mm512_cvtepi8_epi32(yy);
        // __m512i sub = _mm512_sub_epi32(xx_ext, yy_ext);
        // __m512i square = _mm512_mullo_epi32(sub, sub);
        // sum = _mm512_add_epi32(square, sum);
        __m256i xx_ext = _mm256_cvtepi8_epi16(xx);
        __m256i yy_ext = _mm256_cvtepi8_epi16(yy);
        __m256i sub = _mm256_sub_epi16(xx_ext, yy_ext);
        sum = _mm256_add_epi32(_mm256_madd_epi16(sub, sub), sum);
    }
    // reduce for 16 int32 integers in AVX512
    // __m256i sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(sum, 0), _mm512_extracti32x8_epi32(sum, 1));
    // __m128i sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh), _mm256_extracti128_si256(sumh, 1));
    // __m128i tmp = _mm_hadd_epi32(sumhh, sumhh);
    __m128i sumh = _mm_add_epi32(_mm256_extracti32x4_epi32(sum, 0), _mm256_extracti32x4_epi32(sum, 1));
    __m128i tmp = _mm_hadd_epi32(sumh, sumh);
    _mm128_print_epi32(tmp);
    return _mm_extract_epi32(tmp, 0) + _mm_extract_epi32(tmp, 1);
}

inline float
L2SqrSQ8_sym_512(const int8_t *x, const int8_t *y, int d) {
    size_t i;
    d = 32;
    __m512i sum = _mm512_setzero_si512();
    for (int i = 0; i < d; i += 32) {
        __m256i xx = _mm256_load_si256((__m256i *)(x + i));
        __m256i yy = _mm256_load_si256((__m256i *)(y + i));
        __m512i xx_ext = _mm512_cvtepi8_epi16(xx);
        __m512i yy_ext = _mm512_cvtepi8_epi16(yy);
        __m512i sub = _mm512_sub_epi16(xx_ext, yy_ext);
        sum = _mm512_add_epi32(_mm512_madd_epi16(sub, sub), sum);
    }
    // reduce for 16 int32 integers in AVX512
    __m256i sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(sum, 0), _mm512_extracti32x8_epi32(sum, 1));
    __m128i sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh), _mm256_extracti128_si256(sumh, 1));
    __m128i tmp = _mm_hadd_epi32(sumhh, sumhh);
    _mm128_print_epi32(tmp);
    return _mm_extract_epi32(tmp, 0) + _mm_extract_epi32(tmp, 1);
}

// 1. (x-y)^2 = x^2 + y^2 - 2xy, 分别算三个项，32个int16，然后再分别加和
static float
L2SqrSQ8_decouple(const int8_t* x, const int8_t* y, int d) {
    int i;
    // (x-y)^2 = x^2 + y^2 - 2xy
    __m512i x_sqr = _mm512_setzero_si512();
    __m512i y_sqr = _mm512_setzero_si512();
    __m512i xy = _mm512_setzero_si512();
    int32_t res = 0;
    d = 32;
    for (int i = 0; i < d; i += 32) {
        __m256i xx = _mm256_loadu_si256((__m256i *)(x + i));
        __m256i yy = _mm256_loadu_si256((__m256i *)(y + i));
        __m512i xx_ext = _mm512_cvtepi8_epi16(xx);
        __m512i yy_ext = _mm512_cvtepi8_epi16(yy);
        __m512i xx_sqr = _mm512_mullo_epi16(xx_ext, xx_ext);
        __m512i yy_sqr = _mm512_mullo_epi16(yy_ext, yy_ext);
        __m512i dot = _mm512_mullo_epi16(xx_ext, yy_ext);
        x_sqr = _mm512_add_epi16(x_sqr, xx_sqr);
        y_sqr = _mm512_add_epi16(y_sqr, yy_sqr);
        xy = _mm512_add_epi16(xy, dot);
    }
    // reduce 
}

float normal_l2(float const *a, float const *b, unsigned dim)
{
    float r = 0;
    for (unsigned i = 0; i < dim; ++i)
    {
        float v = float(a[i]) - float(a[i]);
        v *= v;
        r += v;
    }
    return r;
}

int normal_l2(int8_t const *a, int8_t const *b, unsigned dim)
{
    int r = 0;
    for (unsigned i = 0; i < dim; ++i)
    {
        int v = int(a[i]) - int(b[i]);
        v *= v;
        r += v;
    }
    return r;
}

int main(int argc, char **argv)
{

    int8_t a0[32] = {127, 127, 127, -4, 5, 6, 7, 8, 9, -127, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, -24, 25, 26, 27, 28, 29, 30, 31, 32};
    int8_t a1[32] = {-127, -127, 4, 5, 6, 7, 8, 9, 10, 127, 12, -13, 14, 15, 16, 17, 18, 19, -20, 21, 22, 23, 24, -25, 26, 27, 28, 29, 30, 31, 32, 33};

    L2SqrSQ8_sym_512((int8_t*)a0, (int8_t*)a1, 32);
    printf("%d\n", normal_l2(a0, a1, 32));

    // SquareSum(a0, 32);
    // NormalSquareSum(a0, 32);

    // IPSQ8(a0, a1, 32);
    // NormalIP(a0, a1, 32);

    return 0;
}
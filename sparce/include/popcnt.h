/**
 * @file       generic_popcnt.hpp
 * @brief
 * @date       2021-2-14
 * @author     Peter
 * @copyright
 *      Peter of [ThinkSpirit Laboratory](http://thinkspirit.org/)
 *   of [Nanjing University of Information Science & Technology](http://www.nuist.edu.cn/)
 *   all rights reserved
 */

#include <climits>
#include <type_traits>

static_assert(CHAR_BIT == 8, "not support for machine with CHAR_BIT != 8");

template <typename T, int W>
struct _0x3333_loop;

template <typename T>
struct _0x3333_loop<T, 1> :
	public std::integral_constant<T, 0x33>
{
};

template <typename T, int W>
struct _0x3333_loop :
	public std::integral_constant<
	T,
	_0x3333_loop<T, W - 1>::value + (static_cast<T>(0x33) << (W - 1) * 8)
	>
{
};

template <typename T>
struct _0x3333 :
	public _0x3333_loop<T, sizeof(T)>
{
};

template <typename T, int W>
struct _0x5555_loop;

template <typename T>
struct _0x5555_loop<T, 1> :
	public std::integral_constant<T, 0x55>
{
};

template <typename T, int W>
struct _0x5555_loop :
	public std::integral_constant<
	T,
	_0x5555_loop<T, W - 1>::value + (static_cast<T>(0x55) << (W - 1) * 8)
	>
{
};

template <typename T>
struct _0x5555 :
	public _0x5555_loop<T, sizeof(T)>
{
};



template <typename T, int W>
struct _0x0f0f_loop;

template <typename T>
struct _0x0f0f_loop<T, 1> :
	public std::integral_constant<T, 0x0f>
{
};

template <typename T, int W>
struct _0x0f0f_loop :
	public std::integral_constant<
	T,
	_0x0f0f_loop<T, W - 1>::value + (static_cast<T>(0xf) << (W - 1) * 8)
	>
{
};

template <typename T>
struct _0x0f0f :
	public _0x0f0f_loop<T, sizeof(T)>
{
};



template <typename T, int W>
struct _0x0101_loop;

template <typename T>
struct _0x0101_loop<T, 1> :
	public std::integral_constant<T, 0x01>
{
};

template <typename T, int W>
struct _0x0101_loop :
	public std::integral_constant<
	T,
	_0x0101_loop<T, W - 1>::value + (static_cast<T>(1) << (W - 1) * 8)
	>
{
};

template <typename T>
struct _0x0101 :
	public _0x0101_loop<T, sizeof(T)>
{
};

template <typename T>
int my_popcount_impl(T x)
{
	x = x - ((x >> 1) & _0x5555<T>::value); // 优化自 swar 算法的 x = (x & 0x55555555) + ((x >> 1) & 0x55555555);
	x = (x & _0x3333<T>::value) + ((x >> 2) & _0x3333<T>::value);
	x = ((x >> 4) + x) & _0x0f0f<T>::value; // 优化自 swar 算法的 x = (x & 0x0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F);
	x = static_cast<unsigned>(x * _0x0101<unsigned>::value) >> (8 * (sizeof(unsigned) - 1));
	return x;
}

template <typename T>
int bitcounts(T x)
{
	return my_popcount_impl(static_cast<typename std::make_unsigned<T>::type>(x));
}

template<>
int bitcounts(bool x)
{
	return x;
}
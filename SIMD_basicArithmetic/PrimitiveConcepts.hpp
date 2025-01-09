#pragma once

#include "PrimitiveTypes.hpp"

#include <type_traits>
#include <concepts>

template <typename T>
concept Integral = std::is_integral<T>::value;

template <typename T>
concept Floating = std::is_floating_point<T>::value;

template <typename T>
concept Numeric = Floating<T> || Integral<T>;

template <typename T>
concept Integral8 = Integral<T> && sizeof(T) == 1;

template <typename T>
concept Integral16 = Integral<T> && sizeof(T) == 2;

template <typename T>
concept Integral32 = Integral<T> && sizeof(T) == 4;

template <typename T>
concept Integral64 = Integral<T> && sizeof(T) == 8;

template <typename T>
concept Floating32 = Floating<T> && sizeof(T) == 4;

template <typename T>
concept Floating64 = Floating<T> && sizeof(T) == 8;

template <typename T>
concept Numeric32 = Numeric<T> && sizeof(T) == 4;

template <typename T>
concept Numeric64 = Numeric<T> && sizeof(T) == 8;
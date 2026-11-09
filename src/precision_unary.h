#ifndef TULPARATIO_PRECISION_UNARY_H
#define TULPARATIO_PRECISION_UNARY_H
#include <tulpa/precision_math.h>
namespace tulpa { namespace math {
template<> struct PrecisionUnary<fwd::Dual> {
    static fwd::Dual apply(const fwd::Dual& x, double value, double derivative) {
        return fwd::Dual(value, x.grad == 0 ? 0 : derivative*x.grad);
    }
};
template<class T> struct PrecisionArenaUnary {
    static T apply(const T& x, double value, double derivative) {
        T out; out.arena_ = x.arena_;
        out.idx_ = x.arena_->add_unary(value, x.idx_, derivative);
        return out;
    }
};
} }
#endif

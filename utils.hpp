#pragma once

#include <algorithm>

#include "image.hpp"
#include "image_expr.hpp"

namespace utils {
    extern "C" {
        void downsa2d(float *oy, const float *ox, int w, int h, int pd, int n, int ty);
        void zoom2(float *y, const float *x, int W, int H, int pd, int w, int h, float n, int zt);
        void downscale_image(float *y, float *x, int outw, int outh, int inw, int inh, float scale);
    }

    inline void downsample(img_t<float>& out, const img_t<float>& in, int factor) {
        out.resize(in.w / factor, in.h / factor, in.d);
        downsa2d(&out[0], &in[0], in.w, in.h, in.d, factor, 'v');
    }

    inline void upsample(img_t<float>& out, const img_t<float>& in, float factor, int targetw, int targeth, int interp=2/* bilinear */) {
        out.resize(targetw, targeth, in.d);
        zoom2(&out[0], &in[0], out.w, out.h, out.d, in.w, in.h, factor, interp);
    }

    void gaussian_downsample(img_t<float>& out, const img_t<float>& in, float factor/* >= 1 */) {
        if (factor == 1) {
            out = in;
            return;
        }
        if (out.size == 0) {
            out.resize(std::ceil(in.w/factor), std::ceil(in.h/factor), in.d);
        }

        img_t<float> tmpout(out.w, out.h);
        img_t<float> tmpin(in.w, in.h);
        for (int d = 0; d < in.d; d++) {
            tmpin.map(slice(in, _, _, _(d)));
            downscale_image(&tmpout[0], &tmpin[0], tmpout.w, tmpout.h, tmpin.w, tmpin.h, factor);
            slice(out, _, _, _(d)).map(tmpout);
        }
    }

    template <typename T>
    T getpixel_1(const img_t<T>& x, int i, int j, int d=0)
    {
        i = std::max(std::min(i, x.w - 1), 0);
        j = std::max(std::min(j, x.h - 1), 0);
        return x(i, j, d);
    }

    template <typename T>
    void downsa2(img_t<T>& out, const img_t<T>& in)
    {
        if (out.size == 0)
            out.resize(in.w/2, in.h/2, in.d);
        for (int d = 0; d < out.d; d++)
        for (int j = 0; j < out.h; j++)
        for (int i = 0; i < out.w; i++)
        {
            T m = getpixel_1(in, 2*i, 2*j, d)
                + getpixel_1(in, 2*i+1, 2*j, d)
                + getpixel_1(in, 2*i, 2*j+1, d)
                + getpixel_1(in, 2*i+1, 2*j+1, d);
            out(i, j, d) = m / T(4);
        }
    }

    template <typename T>
    T evaluate_bilinear_cell(T a[4], float x, float y)
    {
        T r = 0;
        r += a[0] * (1-x) * (1-y);
        r += a[1] * ( x ) * (1-y);
        r += a[2] * (1-x) * ( y );
        r += a[3] * ( x ) * ( y );
        return r;
    }

    template <typename T>
    T bilinear_interpolation(const img_t<T>& x, float p, float q, int d)
    {
        int ip = floor(p);
        int iq = floor(q);
        T a[4] = {
            getpixel_1(x, ip  , iq  , d),
            getpixel_1(x, ip+1, iq  , d),
            getpixel_1(x, ip  , iq+1, d),
            getpixel_1(x, ip+1, iq+1, d)
        };
        T r = evaluate_bilinear_cell(a, p-ip, q-iq);
        return r;
    }

    template <typename T>
    void upsa2(img_t<T>& out, const img_t<T>& in)
    {
        if (out.size == 0)
            out.resize(in.w*2, in.h*2, in.d);
        for (int d = 0; d < out.d; d++)
        for (int j = 0; j < out.h; j++)
        for (int i = 0; i < out.w; i++)
        {
            float x = (i - 0.5) / 2;
            float y = (j - 0.5) / 2;
            out(i, j, d) = bilinear_interpolation(in, x, y, d);
        }
    }

    template <typename T>
    void med(const img_t<T>& _data, T& med)
    {
        auto data(_data);
        std::sort(&data.data[0], &data.data[0] + data.size);
        med = data[data.size / 2];
    }

    template <typename T>
    void medmad(const img_t<T>& _data, T& med, T& mad)
    {
        auto data(_data);
        utils::med(data, med);
        data.map(std::abs(data - med));
        utils::med(data, mad);
    }

    template <typename T>
    void boxfilter(img_t<T>& img, int width)
    {
        if (width == img.w || width == img.h) {
            img.set_value(img::sum<T>(img) / (width*width));
            return;
        }

        int hw = width / 2;

        img_t<T> tmp;
        tmp.resize(img);
        for (int d = 0; d < img.d; d++) {
            for (int y = 0; y < img.h; y++) {
                double v = img(0, y, d);
                for (int x = 1; x <= hw; x++) {
                    v += img(x, y, d) + img(img.w - x, y, d);
                }

                tmp(0, y, d) = v;
                for (int x = 1; x <= hw; x++) {
                    v += img(x + hw, y, d) - img(img.w + x - hw - 1, y, d);
                    tmp(x, y, d) = v;
                }
                for (int x = hw + 1; x < img.w - hw; x++) {
                    v += img(x + hw, y, d) - img(x - hw - 1, y, d);
                    tmp(x, y, d) = v;
                }
                for (int x = img.w - hw; x < img.w; x++) {
                    v += img(x - img.w + hw, y, d) - img(x - hw - 1, y, d);
                    tmp(x, y, d) = v;
                }
            }

            for (int x = 0; x < img.w; x++) {
                double v = tmp(x, 0, d);
                for (int y = 1; y <= hw; y++) {
                    v += tmp(x, y, d) + tmp(x, img.h - y, d);
                }

                img(x, 0, d) = v;
                for (int y = 1; y <= hw; y++) {
                    v += tmp(x, y + hw, d) - tmp(x, img.h + y - hw - 1, d);
                    img(x, y, d) = v;
                }
                for (int y = hw + 1; y < img.h - hw; y++) {
                    v += tmp(x, y + hw, d) - tmp(x, y - hw - 1, d);
                    img(x, y, d) = v;
                }
                for (int y = img.h - hw; y < img.h; y++) {
                    v += tmp(x, y - img.h + hw, d) - tmp(x, y - hw - 1, d);
                    img(x, y, d) = v;
                }
            }
        }

        img.map(img / (width*width));
    }

    /// convert an image to YCbCr colorspace (from RGB)
    template <typename T>
    void rgb2ycbcr(img_t<T>& out, const img_t<T>& in)
    {
        assert(in.d == 3);

        out.resize(in);
        for (int i = 0; i < out.w*out.h; i++) {
            T r = in[i*3+0];
            T g = in[i*3+1];
            T b = in[i*3+2];
            out[i*3+0] = 0.299*r + 0.587*g + 0.114*b;
            out[i*3+1] = (b - out[i*3+0]) * 0.564 + 0.5;
            out[i*3+2] = (r - out[i*3+0]) * 0.713 + 0.5;
        }
    }

    /// convert an image to RGB colorspace (from YCbCr)
    template <typename T>
    void ycbcr2rgb(img_t<T>& out, const img_t<T>& in)
    {
        assert(in.d == 3);

        out.resize(in);
        for (int i = 0; i < out.w*out.h; i++) {
            T y = in[i*3+0];
            T cb = in[i*3+1];
            T cr = in[i*3+2];
            out[i*3+0] = y + 1.403 * (cr - 0.5);
            out[i*3+1] = y - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5);
            out[i*3+2] = y + 1.773 * (cb - 0.5);
        }
    }

    template <typename T>
    void transpose(img_t<T>& out, const img_t<T>& in)
    {
        if (&in == &out) {
            auto copy = in;
            return transpose(out, copy);
        }

        out.resize(in.h, in.w, in.d);
        for (int d = 0; d < in.d; d++) {
            for (int y = 0; y < in.h; y++) {
                for (int x = 0; x < in.w; x++) {
                    out(y, x, d) = in(x, y, d);
                }
            }
        }
    }

    template <typename T>
    img_t<T> add_padding(const img_t<T>& _f, int hw, int hh)
    {
        img_t<T> f(_f.w + hw*2, _f.h + hh*2, _f.d);
        f.set_value(T(0));
        slice(f, _(hw, -hw-1), _(hh, -hh-1)).map(_f);
        // replicate borders
        for (int y = 0; y < hh; y++) {
            for (int x = 0; x < f.w; x++) {
                for (int l = 0; l < f.d; l++) {
                    f(x, y, l) = f(x, 2*hh - y, l);
                    f(x, f.h-1-y, l) = f(x, f.h-1-2*hh+y, l);
                }
            }
        }
        for (int y = 0; y < f.h; y++) {
            for (int x = 0; x < hw; x++) {
                for (int l = 0; l < f.d; l++) {
                    f(x, y, l) = f(2*hw - x, y, l);
                    f(f.w-1-x, y, l) = f(f.w-1-2*hw+x, y, l);
                }
            }
        }
        return f;
    }

    template <typename T>
    img_t<T> add_padding(const img_t<T>& f, const img_t<T>& K)
    {
        return add_padding(f, K.w/2, K.h/2);
    }

    template <typename T>
    img_t<T> remove_padding(const img_t<T>& f, int hw, int hh)
    {
        return to_img(slice(f, _(hw, -hw-1), _(hh, -hh-1)));
    }

    template <typename T>
    img_t<T> remove_padding(const img_t<T>& f, const img_t<T>& K)
    {
        return remove_padding(f, K.w/2, K.h/2);
    }

}


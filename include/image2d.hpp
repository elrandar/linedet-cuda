#pragma once

#include <vector>
#include <iostream>

namespace kalman
{
    class image_point
    {
    public:
        int x;
        int y;
        image_point(int x, int y) : x(x), y(y)
        {
        }
    };

    template <typename T>
    class image2d
    {
    private:
        std::vector<T> buffer;

        T &access_elm(int x, int y)
        {
            if (x >= height || y >= width)
                throw std::invalid_argument("Not in size of image lol");

            return buffer[x * width + y];
        };

    public:
        size_t height;
        size_t width;

        image2d(/* args */);
        image2d(int width, int height)
            : height(height), width(width), buffer(std::vector<T>(width * height))
        {
        }

        image2d(std::initializer_list<std::initializer_list<T>> init_list)
        {
            height = init_list.size();
            width = init_list.begin()->size();
            for (auto list : init_list)
            {
                for (T elm : list)
                {
                    buffer.push_back(elm);
                }
            }
            std::cout << height << " -- " << width;
        }

        T &operator()(std::initializer_list<int> list)
        {
            if (list.size() != 2)
            {
                throw std::invalid_argument("Not a image_point lol");
            }
            int x = *(list.begin());
            int y = *(list.begin() + 1);
            return access_elm(x, y);
        }

        T &operator()(image_point pt)
        {
            return access_elm(pt.x, pt.y);
        }

        ~image2d(){};

        size_t size() const
        {
            return width;
        }
        size_t size(int i) const
        {
            if (i == 0)
                return width;
            else
                return height;
        }

        image2d<T> copy()
        {
            auto out = image2d(width, height);
            auto out_buf = out.get_buffer();

            for (int i = 0; i < buffer.size(); i++)
            {
                out_buf[i] = buffer[i];
            }
            return out;
        }

        std::vector<image_point> domain()
        {
            auto out = std::vector<image_point>();

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    out.push_back(image_point(i, j));
                }
            }
            return out;
        }

        T at(std::initializer_list<int> list)
        {
            return (*this)(list);
        }
        std::vector<T> &get_buffer()
        {
            return buffer;
        }

        std::vector<T> get_buffer_const() const
        {
            return buffer;
        }

        void set_buffer(std::vector<T> b)
        {
            if (b.size() != height * width)
                throw std::invalid_argument("The buffer should have the right size.");
            for (int i = 0; i < b.size(); i++)
            {
                buffer[i] = b[i];
            }
        }

        void fill(T val)
        {
            for (auto &elm : buffer)
                elm = val;
        }

        typedef T (*transform_func)(T);

        image2d<T> &transform(transform_func fun)
        {
            for (auto &elm : buffer)
                elm = fun(elm);
            return *this;
        }
    };

    template <typename T>
    using transform_2_func = T (*)(T, T);

    template <typename T>
    image2d<T> &transform(image2d<T> in_1, image2d<T> in_2, image2d<T> out, transform_2_func<T> f)
    {
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &out, const image2d<T> &img)
    {
        std::vector<T> buffer = img.get_buffer_const();

        out << "[\n[";
        for (size_t i = 0; i < buffer.size(); i++)
        {

            if (i % img.width == 0 && i != 0)
            {
                out << ']' << std::endl
                    << '[' << buffer[i] << ',';
            }
            else
                out << buffer[i] << ',';
        }
        out << "]\n]";
        out << std::endl;
        return out;
    }
}
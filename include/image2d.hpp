#pragma once

#include <vector>
#include <iostream>

class point
{
    public:
    int x;
    int y;
    point(int x, int y): x(x), y(y)
    {}
};

std::ostream& operator<<(std::ostream& out, const point& p)
{
    out << "(" << p.x << "," << p.y << ")";
    return out;
}

template <typename T>
class image2d
{
private:
    std::vector<T> buffer;


    T& access_elm(int x, int y)
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
    {}

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


    T& operator()(std::initializer_list<int> list)
    {
        if (list.size() != 2)
        {
            throw std::invalid_argument("Not a point lol");
        }
        int x = *(list.begin());
        int y = *(list.begin() + 1);
        return access_elm(x, y);
    }

    T& operator()(point pt)
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

    std::vector<point> domain()
    {
        auto out = std::vector<point>();

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                out.push_back(point(i, j));
            }
        }
        return out;
    }

    T at(std::initializer_list<int> list)
    {
        return *this(list);
    }
    std::vector<T> get_buffer() const
    {
        return buffer;
    }

    void fill(T val)
    {
        for (auto & elm : buffer)
            elm = val;
    }

    void transform()
};

template <typename T>
std::ostream &operator<<(std::ostream &out, const image2d<T>& img)
{
    std::vector<T> buffer = img.get_buffer();

    out << "[\n[";
    for (size_t i = 0; i < buffer.size(); i++)
    {
        
        if (i % img.width == 0 && i != 0)
        {
            out << ']' << std::endl << '[' << buffer[i] << ',';
        }
        else
            out << buffer[i] << ',';
    }
    out << "]\n]";
    out << std::endl;
    return out;
}
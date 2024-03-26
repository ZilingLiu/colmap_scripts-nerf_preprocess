from rembg import remove

input_path = '/hhd2/home/Code/lzl_gen/colmap_scripts-nerf_preprocess/scripts/test1.png'
output_path = 'test2.png'

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)
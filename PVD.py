from PIL import Image
import numpy as np

def calculate_psnr(original_path, modified_path, mode): 

    original = Image.open(original_path).convert(mode)
    modified = Image.open(modified_path).convert(mode)

    original_array = np.array(original)
    modified_array = np.array(modified)

    mse = np.mean((original_array - modified_array) ** 2)
   
    if mse == 0:
        return float('inf')

    max_pixel = 255.0 
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def calculate_average_capacity(image_path):
    with Image.open(image_path).convert('L') as img:
        pixels = list(img.getdata())
        
        total_bits = 0
        total_pairs = 0

        for i in range(0, len(pixels) - 1, 2):
            p1 = pixels[i]
            p2 = pixels[i + 1]
            diff = abs(p1 - p2)
            n = thresholding(diff)
            
            total_bits += n
            total_pairs += 1

        average_capacity = total_bits / total_pairs if total_pairs > 0 else 0
        return average_capacity


def message_to_binary(message):
    return ''.join(format(ord(char), '08b') for char in message)


def thresholding(diff):
    if diff <= 8:
        return 3
    elif diff <= 16:
        return 3
    elif diff <= 32:
        return 4
    elif diff <= 64:
        return 5
    elif diff <= 128:
        return 6
    elif diff <= 255:
        return 7
    else:
        return 0


def embed_image(message, cover_path, output_path):

    message_binaries = message_to_binary(message) 
    message_index = 0

    with Image.open(cover_path) as img:
        gray_img = img.convert('L')  
        
        pixels = list(gray_img.getdata())  
        modified_pixels = list(pixels)  

        for i in range(0, len(pixels) - 1, 2):
            p1 = pixels[i]
            p2 = pixels[i + 1]
    

            n = thresholding(p1 - p2)

            if message_index < len(message_binaries):
              
                bits_to_embed = message_binaries[message_index:message_index + n]
                if len(bits_to_embed) < n:
                    break 

                b = int(bits_to_embed, 2)

             
                mask = (1 << n) - 1  
                new_p1 = (p1 & ~mask) | (b & mask) 

      
                modified_pixels[i] = new_p1

                message_index += n


        modified_img = Image.new('L', gray_img.size)
        modified_img.putdata(modified_pixels)

        modified_img.save(output_path)


def extract_message(stego_path):
    extracted_bits = []

    with Image.open(stego_path) as img:
        gray_img = img.convert('L')
        pixels = list(gray_img.getdata())

        for i in range(0, len(pixels) - 1, 2):
            p1 = pixels[i]
            p2 = pixels[i + 1]


            diff = abs(p1 - p2)
            n = thresholding(diff)

            mask = (1 << n) - 1  
            extracted_bits_segment = p1 & mask 
   
            binary_segment = format(extracted_bits_segment, f'0{n}b')
            extracted_bits.append(binary_segment)

    extracted_message = ''.join(extracted_bits)

    message = ''
    for i in range(0, len(extracted_message), 8):
        byte = extracted_message[i:i + 8]
        if byte:
            message += chr(int(byte, 2))

    return message




# message = "This is a sample message!"
message = "Hello, World!"
orig_image = "lena.tiff"
output_image = "thisismyoutput.tiff"

embed_image(message, orig_image, output_image)

print(calculate_psnr(orig_image, output_image, "L"))

print(calculate_average_capacity(orig_image))

print(extract_message(output_image)[:len(message)])

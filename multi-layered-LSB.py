import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt


def message_to_binary(message):
    """Convert text message to binary string"""
    return ''.join(format(ord(char), '08b') for char in message)

def binary_to_message(binary_string):
    """Convert binary string back to text message"""
    # Extract 8 bits at a time and convert to character
    message = ''.join(chr(int(binary_string[i:i+8], 2)) 
                     for i in range(0, len(binary_string), 8))
    return message

def embed_message(cover_path, message, output_path, rounds=1):
    """Embed message using multiple rounds of LSB embedding (max 3 rounds per channel)"""
    # Validate rounds parameter
    if not 1 <= rounds <= 3:
        raise ValueError("Number of rounds must be between 1 and 3")
        
    # Convert message to binary
    binary_message = message_to_binary(message)
    
    # Load and prepare cover image
    cover = Image.open(cover_path)
    is_rgb = cover.mode == 'RGB'
    if not is_rgb:
        cover = cover.convert('L')  # Convert to grayscale
    cover_array = np.array(cover)
    
    # Calculate embedding capacity based on rounds and channels
    channels = 3 if is_rgb else 1
    max_bits = cover_array.size * rounds * channels
    if len(binary_message) > max_bits:
        raise ValueError("Message too long for cover image capacity")
    
    # Prepare array for processing
    if is_rgb:
        # Process each RGB channel separately
        bit_index = 0
        for channel in range(3):  # R, G, B channels
            channel_data = cover_array[..., channel].flatten()
            for round_num in range(rounds):
                for i in range(len(channel_data)):
                    if bit_index >= len(binary_message):
                        break
                    
                    pixel = int(channel_data[i])
                    # Clear specific LSB for this round
                    mask = ~(1 << round_num)
                    pixel = pixel & mask
                    
                    # Get next message bit
                    if bit_index < len(binary_message):
                        message_bit = int(binary_message[bit_index])
                        # Embed bit at appropriate position based on round
                        pixel = pixel | (message_bit << round_num)
                        channel_data[i] = pixel
                        bit_index += 1
            
            cover_array[..., channel] = channel_data.reshape(cover_array[..., channel].shape)
    else:
        # Process grayscale channel
        flat_image = cover_array.flatten()
        bit_index = 0
        for round_num in range(rounds):
            for i in range(len(flat_image)):
                if bit_index >= len(binary_message):
                    break
                
                pixel = int(flat_image[i])
                mask = ~(1 << round_num)
                pixel = pixel & mask
                
                if bit_index < len(binary_message):
                    message_bit = int(binary_message[bit_index])
                    pixel = pixel | (message_bit << round_num)
                    flat_image[i] = pixel
                    bit_index += 1
        
        cover_array = flat_image.reshape(cover_array.shape)
    
    # Save stego image
    stego_image = Image.fromarray(cover_array.astype(np.uint8))
    stego_image.save(output_path, format='PNG')
    
    return stego_image

def extract_message(stego_path, rounds=1):
    """Extract message from stego image using multiple rounds"""
    if not 1 <= rounds <= 3:
        raise ValueError("Number of rounds must be between 1 and 3")
        
    # Load stego image
    stego = Image.open(stego_path)
    is_rgb = stego.mode == 'RGB'
    if not is_rgb:
        stego = stego.convert('L')  # Convert to grayscale
    stego_array = np.array(stego)
    
    # Extract LSBs from each pixel for each round and channel
    extracted_bits = []
    
    if is_rgb:
        # Extract from each RGB channel
        for channel in range(3):
            channel_data = stego_array[..., channel].flatten()
            for round_num in range(rounds):
                for pixel in channel_data:
                    bit = (pixel >> round_num) & 1
                    extracted_bits.append(str(bit))
    else:
        # Extract from grayscale channel
        flat_stego = stego_array.flatten()
        for round_num in range(rounds):
            for pixel in flat_stego:
                bit = (pixel >> round_num) & 1
                extracted_bits.append(str(bit))
    
    # Join bits and convert to message
    binary_message = ''.join(extracted_bits)
    try:
        message = binary_to_message(binary_message)
        return message
    except:
        return ""

def calculate_psnr(original_path, stego_path):
    """Calculate PSNR between original and stego images"""
    original = np.array(Image.open(original_path))
    stego = np.array(Image.open(stego_path))
    
    mse = np.mean((original - stego) ** 2)
    if mse == 0:
        return float('inf')
        
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_capacity(image_path, rounds=1):
    """Calculate maximum capacity in bytes based on image size, channels and embedding rounds"""
    img = Image.open(image_path)
    is_rgb = img.mode == 'RGB'
    if not is_rgb:
        img = img.convert('L')
    channels = 3 if is_rgb else 1
    total_pixels = np.array(img).size
    max_bits = total_pixels * rounds * channels
    max_bytes = max_bits // 8
    return max_bytes

def calculate_bpp(message, image_path, rounds):
    """Calculate bits per pixel (BPP) for the embedding"""
    binary_message = message_to_binary(message)
    img = Image.open(image_path)
    is_rgb = img.mode == 'RGB'
    if not is_rgb:
        img = img.convert('L')
    total_pixels = np.array(img).size
    channels = 3 if is_rgb else 1
    bpp = (len(binary_message) * rounds * channels) / total_pixels
    return bpp

if __name__ == "__main__":
    # Test the implementation
    cover_image = "test-images/lena.tiff"
    stego_image = "stego.png"

    message_file = "payload1.txt"
    with open(message_file, "r") as f:
        message = f.read().strip()

    rounds = 3
    
    try:
        
        # Embed message using 2 rounds of LSB
        embed_message(cover_image, message, stego_image, rounds=rounds)
        print("Message embedded successfully")

        # Get message length in bytes
        message_length = len(message.encode('utf-8'))
        print(f"Message length: {message_length} bytes")


        # Extract message using same number of rounds
        extracted = extract_message(stego_image, rounds=rounds)
        print(f"Extracted message: {extracted[-10:]}")
        
        # Calculate PSNR
        psnr = calculate_psnr(cover_image, stego_image)
        print(f"PSNR: {psnr:.2f} dB")

        # Calculate and display capacity and BPP
        max_capacity = calculate_capacity(cover_image, rounds)
        bpp = calculate_bpp(message, cover_image, rounds)
        print(f"Maximum capacity: {max_capacity} bytes")
        print(f"Bits per pixel (BPP): {bpp:.4f}")

        # Display original and stego images side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(Image.open(cover_image))
        ax1.set_title('Original Image')
        ax1.axis('off')
        ax2.imshow(Image.open(stego_image))
        ax2.set_title('Stego Image')
        ax2.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")

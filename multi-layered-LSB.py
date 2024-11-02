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
    """Embed message using multiple rounds of LSB embedding (max 3 rounds)"""
    # Validate rounds parameter
    if not 1 <= rounds <= 3:
        raise ValueError("Number of rounds must be between 1 and 3")
        
    # Convert message to binary
    binary_message = message_to_binary(message)
    
    # Load and prepare cover image
    cover = Image.open(cover_path)
    if cover.mode != 'RGB':
        cover = cover.convert('RGB')
    cover_array = np.array(cover)
    
    # Calculate embedding capacity based on rounds
    max_bits = cover_array.size * rounds
    if len(binary_message) > max_bits:
        raise ValueError("Message too long for cover image capacity")
        
    # Flatten array for easier processing
    flat_image = cover_array.flatten()
    
    bit_index = 0
    # Embed bits across pixels using multiple rounds
    for round_num in range(rounds):
        for i in range(len(flat_image)):
            if bit_index >= len(binary_message):
                break
                
            pixel = int(flat_image[i])
            # Clear specific LSB for this round
            mask = ~(1 << round_num)
            pixel = pixel & mask
            
            # Get next message bit
            if bit_index < len(binary_message):
                message_bit = int(binary_message[bit_index])
                # Embed bit at appropriate position based on round
                pixel = pixel | (message_bit << round_num)
                flat_image[i] = pixel
                bit_index += 1
    
    # Reshape and save stego image
    stego = flat_image.reshape(cover_array.shape)
    stego_image = Image.fromarray(stego.astype(np.uint8))
    stego_image.save(output_path, format='PNG')
    
    return stego_image

def extract_message(stego_path, rounds=1):
    """Extract message from stego image using multiple rounds"""
    if not 1 <= rounds <= 3:
        raise ValueError("Number of rounds must be between 1 and 3")
        
    # Load stego image
    stego = Image.open(stego_path)
    if stego.mode != 'RGB':
        stego = stego.convert('RGB')
    stego_array = np.array(stego)
    
    # Extract LSBs from each pixel for each round
    extracted_bits = []
    flat_stego = stego_array.flatten()
    
    for round_num in range(rounds):
        for pixel in flat_stego:
            # Extract bit from appropriate position based on round
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
    """Calculate maximum capacity in bytes based on image size and embedding rounds"""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    total_pixels = np.array(img).size
    max_bits = total_pixels * rounds
    max_bytes = max_bits // 8
    return max_bytes

def calculate_bpp(message, image_path, rounds):
    """Calculate bits per pixel (BPP) for the embedding"""
    binary_message = message_to_binary(message)
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    total_pixels = np.array(img).size
    bpp = (len(binary_message) * rounds) / total_pixels
    return bpp

if __name__ == "__main__":
    # Test the implementation
    cover_image = "test-images/peppers.tiff"
    stego_image = "stego.png"

    message = "This is a test message for multi-layered LSB steganography!"
    rounds = 3
    
    try:

        
        # Embed message using 2 rounds of LSB
        embed_message(cover_image, message, stego_image, rounds=rounds)
        print("Message embedded successfully")
        
        # Extract message using same number of rounds
        extracted = extract_message(stego_image, rounds=rounds)
        print(f"Extracted message: {extracted[:len(message)]}")
        
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

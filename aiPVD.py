import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import sobel
import math

def detect_edge_regions(image_array):
    """Detect edge regions using Sobel operator and return binary edge map"""
    # Convert to grayscale if color image
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
        
    # Calculate Sobel gradients
    sobel_x = sobel(gray, axis=0)
    sobel_y = sobel(gray, axis=1)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Threshold to get binary edge map
    threshold = gradient_magnitude.mean()
    edge_map = (gradient_magnitude > threshold).astype(int)
    
    return edge_map

def embed_message(cover_path, message, output_path):
    """Embed message in cover image using LSB steganography in edge regions"""
    # Load cover image
    cover = Image.open(cover_path)
    cover_array = np.array(cover)
    
    # Get edge map
    edge_map = detect_edge_regions(cover_array)
    
    # Convert message to binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    message_length = len(binary_message)
    binary_length = format(message_length, '032b')  # 32 bits for length
    binary_data = binary_length + binary_message
    
    # Flatten arrays for easier processing
    flat_image = cover_array.flatten()
    flat_edges = np.repeat(edge_map.flatten(), 3)  # Repeat for RGB channels
    
    bit_pointer = 0
    max_bits = len(binary_data)
    
    # Embed data only in edge pixels
    for i in range(len(flat_image)):
        if bit_pointer >= max_bits:
            break
            
        if flat_edges[i]:  # Only embed in edge pixels
            # Get next bit from message
            message_bit = binary_data[bit_pointer]
            
            # Clear LSB and embed new bit
            pixel = int(flat_image[i])  # Convert to int first
            pixel = (pixel & 254)  # Clear LSB using 254 (0b11111110)
            pixel = pixel | int(message_bit)  # Embed bit
            flat_image[i] = pixel
            
            bit_pointer += 1
    
    # Reshape and save
    stego = flat_image.reshape(cover_array.shape)
    stego_image = Image.fromarray(stego.astype(np.uint8))
    stego_image.save(output_path)
    return stego_image

def extract_message(stego_path):
    """Extract message from stego image"""
    # Load stego image
    stego = Image.open(stego_path)
    stego_array = np.array(stego)
    
    # Get edge map
    edge_map = detect_edge_regions(stego_array)
    
    # Flatten arrays
    flat_stego = stego_array.flatten()
    flat_edges = np.repeat(edge_map.flatten(), 3)
    
    extracted_bits = []
    
    # Extract length first (32 bits)
    bit_count = 0
    i = 0
    while bit_count < 32:
        if flat_edges[i]:
            bit = str(flat_stego[i] & 1)  # Get LSB
            extracted_bits.append(bit)
            bit_count += 1
        i += 1
    
    # Get message length
    length = int(''.join(extracted_bits[:32]), 2)
    extracted_bits = []
    
    # Extract message bits
    while len(extracted_bits) < length:
        if flat_edges[i]:
            bit = str(flat_stego[i] & 1)
            extracted_bits.append(bit)
        i += 1
    
    # Convert bits to message
    message = ''
    for i in range(0, len(extracted_bits), 8):
        byte = ''.join(extracted_bits[i:i+8])
        message += chr(int(byte, 2))
    
    return message

if __name__ == "__main__":
    # Example usage
    cover_path = "test-images/lena.png"
    secret_message = "This is a secret message using edge-based LSB steganography!"
    stego_path = "stego_lena_edge.png"
    
    # Embed message
    embed_message(cover_path, secret_message, stego_path)
    
    # Extract message
    extracted_message = extract_message(stego_path)
    print("Extracted message:", extracted_message[:200])

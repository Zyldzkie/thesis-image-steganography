import numpy as np
from PIL import Image
import cv2
import math
from cv2 import Canny
import os

def message_to_binary(message):
    """Convert a text message to a binary string"""
    # Convert each character to 8-bit binary and join
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    return binary_message


def binary_to_message(binary_string):
    """Convert a binary string to a text message"""
    # Extract and join message
    message = ''.join(chr(int(binary_string[i:i+8], 2)) for i in range(0, len(binary_string), 8))
    return message

# Test cases
def test_message_conversion():
    # Test empty message
    assert message_to_binary("") == ""
    assert binary_to_message("") == ""
    
    # Test single character
    msg = "A"
    binary = message_to_binary(msg)
    assert len(binary) == 8  # 8-bit char
    assert binary == "01000001"  # ASCII for 'A'
    print(binary_to_message(binary))
    assert binary_to_message(binary) == msg
    
    # Test longer message
    msg = "Hello, World!"
    binary = message_to_binary(msg)
    assert len(binary) == (13 * 8)  # 13 chars
    assert binary_to_message(binary) == msg
    
    # Test special characters
    msg = "!@#$%^&*()"
    binary = message_to_binary(msg)
    assert binary_to_message(binary) == msg  # Test direct conversion
    
    print("All tests passed!")



def detect_edge_regions(image_array, threshold1=50, threshold2=255):
    """Detect edge regions using Canny operator and return binary edge map"""
    # Convert to grayscale if color image
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
        
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Apply Canny edge detection with consistent thresholds
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    # Dilate edges slightly to create more stable regions
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Normalize to binary (0 and 1) values
    edge_map = edges.astype(bool).astype(int)
    
    return edge_map
 

def embed_message(cover_path, message, output_path):
    """Embed message in cover image using LSB steganography in edge regions"""

    message_binary = message_to_binary(message) 
    
    # Load cover image
    cover = Image.open(cover_path)
    # Convert to RGB mode if needed
    if cover.mode != 'RGB':
        cover = cover.convert('RGB')
    cover_array = np.array(cover)

    # Use consistent edge detection parameters
    edge_map = detect_edge_regions(cover_array, threshold1=50, threshold2=255)

    # Flatten arrays for easier processing
    flat_image = cover_array.flatten()
    flat_edges = np.repeat(edge_map.flatten(), 3)  # Repeat for RGB channels
    
    bit_pointer = 0
    max_bits = len(message_binary)
    
    # Store number of edge pixels at start of image
    num_edge_pixels = np.sum(flat_edges)
    if num_edge_pixels * 3 < max_bits:
        raise ValueError("Not enough edge pixels to embed the message")
    
    # Embed data only in edge pixels
    for i in range(len(flat_image)):
        if bit_pointer >= max_bits:
            break
            
        if flat_edges[i]:  # Only embed in edge pixels
            # Get next bit from message
            message_bit = message_binary[bit_pointer]
            
            # Clear LSB and embed new bit
            pixel = int(flat_image[i])
            pixel = (pixel & 254)  # Clear LSB using 254 (0b11111110)
            pixel = pixel | int(message_bit)  # Embed bit
            flat_image[i] = pixel
            
            bit_pointer += 1
    
    # Reshape and save
    stego = flat_image.reshape(cover_array.shape)
    stego_image = Image.fromarray(stego.astype(np.uint8))
    
    # Save with appropriate format
    if output_path.lower().endswith('.tiff') or output_path.lower().endswith('.tif'):
        stego_image.save(output_path, format='TIFF', compression='raw')
    else:  # Default to PNG
        stego_image.save(output_path, format='PNG')
    return stego_image
    


def extract_message(stego_path):
    """Extract message from stego image"""
    stego = Image.open(stego_path)
    # Convert to RGB mode if needed
    if stego.mode != 'RGB':
        stego = stego.convert('RGB')
    stego_array = np.array(stego)

    # Use consistent edge detection parameters
    edge_map = detect_edge_regions(stego_array, threshold1=50, threshold2=255)

    # Flatten arrays
    flat_stego = stego_array.flatten()
    flat_edges = np.repeat(edge_map.flatten(), 3)  # Repeat for RGB channels
    
    extracted_bits = []
    
    # Extract bits from edge pixels
    for i in range(len(flat_edges)):
        if flat_edges[i]:
            bit = str(flat_stego[i] & 1)  # Get LSB
            extracted_bits.append(bit)
    
    # Convert extracted bits to binary string
    binary_message = ''.join(extracted_bits)
    
    # Convert binary to message
    try:
        message = binary_to_message(binary_message)
        return message
    except:
        return ""




if __name__ == "__main__":
    test_message_conversion()
    
    
    png_test_dir = "png_test"
    test_message = "Hello, this is a test message for steganography!"
    
    # Iterate through all PNG files in the directory
    for filename in os.listdir(png_test_dir):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(png_test_dir, filename)
            output_path = os.path.join(png_test_dir, f"..stego_{filename}")
            
            print(f"\nTesting with image: {filename}")
            print("-" * 50)
            
            # Embed message
            try:
                embed_message(input_path, test_message, output_path)
                print(f"Message embedded successfully in {output_path}")
                
                # Extract and verify
                extracted_message = extract_message(output_path)
                print(f"Extracted message: {extracted_message[:100]}")
                
                # Verify if messages match
                if test_message == extracted_message:
                    print("✓ Success: Embedded and extracted messages match")
                else:
                    print("✗ Error: Messages don't match")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # # Test with both PNG and TIFF
    # orig_image = "test-images/peppers.tiff"
    # stego_image_png = "output.png"
    # stego_image_tiff = "output.tiff"

    # # Test PNG
    # embed_message(orig_image, "Hello, World!", stego_image_png)
    # extracted_message = extract_message(stego_image_png)
    # print("Extracted message from PNG:", extracted_message[:100])

    # # Test TIFF
    # embed_message(orig_image, "Hello, World!", stego_image_tiff)
    # extracted_message = extract_message(stego_image_tiff)
    # print("Extracted message from TIFF:", extracted_message[:100])

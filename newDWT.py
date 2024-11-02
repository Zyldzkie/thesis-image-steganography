from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct

def embed_message(cover_image_path, secret_message, output_path):
    # Load the cover image
    cover = Image.open(cover_image_path)
    cover = np.array(cover)
    
    # Convert secret message to binary
    binary_message = ''.join(format(ord(c), '08b') for c in secret_message)
    message_length = len(binary_message)
    
    # Store message length at beginning
    binary_length = format(message_length, '032b')
    binary_data = binary_length + binary_message
    
    # Process each color channel separately
    stego = np.zeros_like(cover)
    bit_index = 0
    
    block_size = 8
    
    for channel in range(3):  # RGB channels
        height, width = cover[:,:,channel].shape
        # Process 8x8 blocks
        for i in range(0, height-block_size+1, block_size):
            for j in range(0, width-block_size+1, block_size):
                if bit_index >= len(binary_data):
                    break
                    
                # Get current 8x8 block
                block = cover[i:i+block_size, j:j+block_size, channel].astype(float)
                
                # Apply DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Embed in mid-frequency coefficient (4,3)
                if bit_index < len(binary_data):
                    # Quantize to nearest 0.5 to improve robustness
                    if binary_data[bit_index] == '1':
                        dct_block[4,3] = round(dct_block[4,3] * 2) / 2 + 0.25
                    else:
                        dct_block[4,3] = round(dct_block[4,3] * 2) / 2 - 0.25
                    bit_index += 1
                
                # Inverse DCT
                block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                stego[i:i+block_size, j:j+block_size, channel] = block
    
    # Clip values and convert back to uint8
    stego = np.clip(stego, 0, 255).astype(np.uint8)
    
    # Save stego image
    stego_image = Image.fromarray(stego)
    stego_image.save(output_path)
    return stego_image

def extract_message(stego_image_path):
    # Load stego image
    stego = Image.open(stego_image_path)
    stego = np.array(stego)
    
    extracted_bits = []
    block_size = 8
    
    for channel in range(3):
        height, width = stego[:,:,channel].shape
        # Process 8x8 blocks
        for i in range(0, height-block_size+1, block_size):
            for j in range(0, width-block_size+1, block_size):
                # Get current 8x8 block
                block = stego[i:i+block_size, j:j+block_size, channel].astype(float)
                
                # Apply DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Extract from mid-frequency coefficient
                coef = dct_block[4,3]
                # Check if coefficient is closer to positive or negative 0.25
                bit = '1' if (coef % 0.5) > 0 else '0'
                extracted_bits.append(bit)
    
    # First 32 bits represent message length
    message_length = int(''.join(extracted_bits[:32]), 2)
    message_bits = extracted_bits[32:32+message_length]
    
    # Convert bits to characters
    message = ''
    for i in range(0, len(message_bits), 8):
        byte = ''.join(message_bits[i:i+8])
        message += chr(int(byte, 2))
    
    return message

# Example usage:
if __name__ == "__main__":
    # Embedding
    cover_path = "test-images/lena.png"
    secret_message = "what!"
    stego_path = "stego_lena.png"
    
    embed_message(cover_path, secret_message, stego_path)
    
    # Extracting
    extracted_message = extract_message(stego_path)
    print("Extracted message:", extracted_message)

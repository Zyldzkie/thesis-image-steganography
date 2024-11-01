import numpy as np
import pywt
from PIL import Image

def load_image(image_path):
    """Load an image and convert to grayscale."""
    img = Image.open(image_path).convert('L')
    return np.array(img)

def perform_dwt(image_array):
    """Perform DWT on the image using 'haar' wavelet."""
    coeffs = pywt.dwt2(image_array, 'haar')
    return coeffs

def inverse_dwt(coeffs):
    """Reconstruct the image from the DWT coefficients."""
    return pywt.idwt2(coeffs, 'haar')

def message_to_binary(message):
    """Convert a message to its binary representation."""
    return ''.join(format(ord(char), '08b') for char in message)

import numpy as np

def embed_message(coeffs, message):
    cA, (cH, cV, cD) = coeffs
    cA_flat = cA.flatten()


    binary_message = ''.join(format(ord(char), '08b') for char in message)

    for i in range(len(binary_message)):

        int_val = int(cA_flat[i])          
        modified_val = (int_val & ~1) | int(binary_message[i]) 
        cA_flat[i] = float(modified_val)      
    
    cA = cA_flat.reshape(cA.shape)
    return (cA, (cH, cV, cD))

def extract_message(coeffs):
    """Extract the message from the DWT coefficients."""
    cA, (cH, cV, cD) = coeffs
    cA_flat = cA.flatten()

    binary_message = ''.join(str(int(coef) & 1) for coef in cA_flat)

    message = ''
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i + 8]
        if byte == '11111111': 
            break
        if len(byte) == 8:
            message += chr(int(byte, 2))
    
    return message

if __name__ == "__main__":

    original_image_path = "test-images/lake.tiff"  
    output_image_path = "output_image.tiff"

    image_array = load_image(original_image_path)

    coeffs = perform_dwt(image_array)

    message = "What the hell is this??? Why is is this even working bro"
    modified_coeffs = embed_message(coeffs, message)

    modified_image = inverse_dwt(modified_coeffs)

    Image.fromarray(np.uint8(modified_image)).save(output_image_path)

    extracted_message = extract_message(modified_coeffs)
    print("Extracted message:", extracted_message[:len(message)])

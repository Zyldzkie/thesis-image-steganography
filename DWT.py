import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

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


def embed_message(coeffs, message):
    cA, (cH, cV, cD) = coeffs
    
    cH_flat = cH.flatten()
    cV_flat = cV.flatten()
    cD_flat = cD.flatten()

    binary_message = ''.join(format(ord(char), '08b') for char in message) + '11111111'
    message_index = 0

    for i in range(len(cH_flat)):
        if message_index >= len(binary_message):
            break
        int_val = int(cH_flat[i])
        modified_val = (int_val & ~1) | int(binary_message[message_index])  # Embed one bit
        cH_flat[i] = float(modified_val)
        message_index += 1

    for i in range(len(cV_flat)):
        if message_index >= len(binary_message):
            break
        int_val = int(cV_flat[i])
        modified_val = (int_val & ~1) | int(binary_message[message_index])
        cV_flat[i] = float(modified_val)
        message_index += 1

    for i in range(len(cD_flat)):
        if message_index >= len(binary_message):
            break
        int_val = int(cD_flat[i])
        modified_val = (int_val & ~1) | int(binary_message[message_index])
        cD_flat[i] = float(modified_val)
        message_index += 1

    cH = cH_flat.reshape(cH.shape)
    cV = cV_flat.reshape(cV.shape)
    cD = cD_flat.reshape(cD.shape)

    return (cA, (cH, cV, cD))

def extract_message(coeffs):
    """Extract message from high-frequency subbands cH, cV, and cD."""
    _, (cH, cV, cD) = coeffs

    cH_flat = cH.flatten()
    cV_flat = cV.flatten()
    cD_flat = cD.flatten()

    binary_message = []

    for coef in cH_flat:
        binary_message.append(str(int(coef) & 1))

    for coef in cV_flat:
        binary_message.append(str(int(coef) & 1))

    for coef in cD_flat:
        binary_message.append(str(int(coef) & 1))

    binary_message = ''.join(binary_message)
    message = ''
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i + 8]
        if byte == '11111111':  
            break
        if len(byte) == 8:
            message += chr(int(byte, 2))

    return message


def calculate_psnr(original, modified):
    """Calculate the PSNR between the original and modified images."""
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return float('inf') 
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_capacity(coeffs):
    """Calculate the embedding capacity based on DWT coefficients."""
    _, (cH, cV, cD) = coeffs
    
    num_coefficients = cH.size + cV.size + cD.size
    
    capacity_bits = num_coefficients - 8 # not taking into account the 8 bits of termination signal
    capacity_bytes = capacity_bits // 8
    
    return capacity_bytes

def calculate_bpp(message, image_array):
    """Calculate the bits per pixel (BPP) of the embedding process."""
    total_bits_embedded = len(message_to_binary(message))
    total_pixels = image_array.size  
    bpp = total_bits_embedded / total_pixels 
    return bpp


def show_subbands(coeffs):
    """Display the DWT subbands."""
    cA, (cH, cV, cD) = coeffs

    plt.figure(figsize=(12, 12))

    # Show the approximation coefficients
    plt.subplot(2, 2, 1)
    plt.imshow(cA, cmap='gray')
    plt.title('Approximation (LL)')
    plt.axis('off')

    # Show the horizontal detail coefficients
    plt.subplot(2, 2, 2)
    plt.imshow(cH, cmap='gray')
    plt.title('Horizontal Detail (LH)')
    plt.axis('off')

    # Show the vertical detail coefficients
    plt.subplot(2, 2, 3)
    plt.imshow(cV, cmap='gray')
    plt.title('Vertical Detail (HL)')
    plt.axis('off')

    # Show the diagonal detail coefficients
    plt.subplot(2, 2, 4)
    plt.imshow(cD, cmap='gray')
    plt.title('Diagonal Detail (HH)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    original_image_path = "test-images/red.png"  
    output_image_path = "output_image.tiff"

    # Load the original image and perform DWT
    image_array = load_image(original_image_path)
    coeffs = perform_dwt(image_array)

    with open("payload.txt", "r") as f:
        message = f.read().strip()
    
    print("Message length", len(message_to_binary(message)) // 8, "bytes") 

    # Embed the message in the coefficients
    modified_coeffs = embed_message(coeffs, message)

    # Inverse DWT to create the modified image
    modified_image = inverse_dwt(modified_coeffs)

    # Save the modified image
    Image.fromarray(np.uint8(modified_image)).save(output_image_path)


    

    # Load the saved stego image for extraction
    stego_image_array = load_image(output_image_path)

    # Perform DWT on the stego image
    extracted_coeffs = perform_dwt(stego_image_array)

    # Extract the message from the new coefficients
    extracted_message = extract_message(extracted_coeffs)
    print("Extracted message:", extracted_message)

    # Optionally, calculate metrics
    psnr_value = calculate_psnr(image_array, modified_image)
    print("PSNR:", psnr_value)

    capacity_value = calculate_capacity(coeffs)
    print("Max Capacity:", capacity_value, "bytes")

    bpp_value = calculate_bpp(message, image_array)
    print("BPP:", bpp_value)
    
    show_subbands(coeffs=coeffs)



    


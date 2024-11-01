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

    # Convert message to binary with termination signal
    binary_message = ''.join(format(ord(char), '08b') for char in message) + '11111111'
    message_index = 0

    # Function to embed bits into coefficients
    def embed_bits(coeff_flat):
        nonlocal message_index
        for i in range(len(coeff_flat)):
            if message_index < len(binary_message) - 1:  # Ensure there's room for 2 bits
                int_val = int(coeff_flat[i])
                bit1 = int(binary_message[message_index])      # First bit
                bit2 = int(binary_message[message_index + 1])  # Second bit

                # Modify the last two bits of the coefficient
                modified_val = (int_val & ~3) | (bit1 << 1) | bit2  # Embed two bits
                coeff_flat[i] = float(modified_val)

                message_index += 2  # Move to the next two bits
            else:
                break  # Exit if we have processed all bits in the message

    # Embed into cH, cV, cD
    embed_bits(cH_flat)
    embed_bits(cV_flat)
    embed_bits(cD_flat)

    # Reshape back to original
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

    # Function to extract bits from coefficients
    def extract_bits(coeff_flat):
        for coef in coeff_flat:
            # Extract the last two bits
            bits = int(coef) & 3  # Get the last two bits
            # Append the bits as a two-character binary string
            binary_message.append(format(bits, '02b'))

    # Extract from cH, cV, cD
    extract_bits(cH_flat)
    extract_bits(cV_flat)
    extract_bits(cD_flat)

    binary_message = ''.join(binary_message)
    message = ''

    # Process the binary message in pairs
    for i in range(0, len(binary_message), 8):  # Change to process in 8 bits for a byte
        byte = binary_message[i:i + 8]  # Extract 8 bits
        if byte == '11111111':  # Termination signal
            break
        if len(byte) == 8:  # Ensure it’s a complete byte
            byte_value = int(byte, 2)  # Convert to integer
            message += chr(byte_value)  # Convert to character

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
    capacity_bytes = (capacity_bits // 8) * 2
    
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

    original_image_path = "test-images/peppers.tiff"  
    output_image_path = "output_image.tiff"

    image_array = load_image(original_image_path)

    coeffs = perform_dwt(image_array)

    with open("payload1.txt", "r") as f:
        #message = "Hello World!"
        message = f.read().strip()  
    
    # Should be 49152 bytes when 2 bits can be flipped in the 3 sub band
    print("Message length", len(message_to_binary(message))//8, "bytes") # Max Message length 24576 bytes 


    modified_coeffs = embed_message(coeffs, message)

    modified_image = inverse_dwt(modified_coeffs)

    Image.fromarray(np.uint8(modified_image)).save(output_image_path)

    extracted_message = extract_message(modified_coeffs)
    print("Extracted message:", extracted_message[-10:])

    psnr_value = calculate_psnr(image_array, modified_image)
    print("PSNR:", psnr_value)

    capacity_value = calculate_capacity(coeffs)
    print("Max Capacity:", capacity_value, "bytes")

    bpp_value = calculate_bpp(message, image_array)
    print("BPP:", bpp_value)
    
    show_subbands(coeffs=coeffs)


    


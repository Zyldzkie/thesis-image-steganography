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

    def embed_first_lsb(coeff_flat):
        """Embed the first LSB from the binary message into the coefficient array."""
        nonlocal message_index
        for i in range(len(coeff_flat)):
            if message_index < len(binary_message):
                int_val = int(coeff_flat[i])
                bit = int(binary_message[message_index])
                modified_val = (int_val & ~1) | bit
                coeff_flat[i] = float(modified_val)
                message_index += 1

            if message_index >= len(binary_message):
                break

    def embed_second_lsb(coeff_flat):
        """Embed the second LSB from the binary message into the coefficient array."""
        nonlocal message_index
        for i in range(len(coeff_flat)):
            if message_index < len(binary_message):
                int_val = int(coeff_flat[i])
                bit = int(binary_message[message_index])
                modified_val = (int_val & ~2) | (bit << 1)
                coeff_flat[i] = float(modified_val)
                message_index += 1

            if message_index >= len(binary_message):
                break

    def embed_third_lsb(coeff_flat):
        """Embed the third LSB from the binary message into the coefficient array."""
        nonlocal message_index
        for i in range(len(coeff_flat)):
            if message_index < len(binary_message):
                int_val = int(coeff_flat[i])
                bit = int(binary_message[message_index])
                modified_val = (int_val & ~4) | (bit << 2)  # Modify third LSB
                coeff_flat[i] = float(modified_val)
                message_index += 1

            if message_index >= len(binary_message):
                break

    # Embed into cH, cV, cD for the first round of embedding (first LSB)
    embed_first_lsb(cH_flat)
    embed_first_lsb(cV_flat)
    embed_first_lsb(cD_flat)

    # If there's still some message left, continue embedding in the second LSB
    if message_index < len(binary_message):
        embed_second_lsb(cH_flat)
        embed_second_lsb(cV_flat)
        embed_second_lsb(cD_flat)

        if message_index < len(binary_message):
            embed_third_lsb(cH_flat)
            embed_third_lsb(cV_flat)
            embed_third_lsb(cD_flat)

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

    def extract_first_lsb(coeff_flat):
        """Extract the first LSB from the coefficient array."""
        for coef in coeff_flat:
            first_lsb = int(coef) & 1  
            binary_message.append(str(first_lsb))  

    def extract_second_lsb(coeff_flat):
        """Extract the second LSB from the coefficient array."""
        for coef in coeff_flat:
            second_lsb = (int(coef) >> 1) & 1  
            binary_message.append(str(second_lsb))

    def extract_third_lsb(coeff_flat):
        """Extract the third LSB from the coefficient array."""
        for coef in coeff_flat:
            third_lsb = (int(coef) >> 2) & 1  # Extract third LSB
            binary_message.append(str(third_lsb))


    extract_first_lsb(cH_flat)
    extract_first_lsb(cV_flat)
    extract_first_lsb(cD_flat)

    extract_second_lsb(cH_flat)
    extract_second_lsb(cV_flat)
    extract_second_lsb(cD_flat)

    extract_third_lsb(cH_flat)
    extract_third_lsb(cV_flat)
    extract_third_lsb(cD_flat)

    binary_message = ''.join(binary_message)
    message = ''

    for i in range(0, len(binary_message), 8): 
        byte = binary_message[i:i + 8]  
        if byte == '11111111':  
            break
        if len(byte) == 8:  
            byte_value = int(byte, 2)  
            message += chr(byte_value)  

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
    capacity_bytes = (capacity_bits // 8) * 3
    
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


def do_main():
    original_image_path = "test-images/red.png"  
    output_image_path = "output_image.tiff"

    image_array = load_image(original_image_path)

    coeffs = perform_dwt(image_array)

    with open("payload3.txt", "r") as f:
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



def do_separate():

    original_image_path = "test-images/lena.png"  
    output_image_path = "output_image.tiff"
    message_file = "payload1.txt"

    def embed_process(original_image_path, output_image_path, message_file):
        orig_image_array = load_image(original_image_path)

        coeffs = perform_dwt(orig_image_array)

        with open(message_file, "r") as f:
            message = "What the helllll"
            #message = f.read().strip()  
            
        print("Message length", len(message_to_binary(message)) // 8, "bytes") 

        modified_coeffs = embed_message(coeffs, message)

        modified_image = inverse_dwt(modified_coeffs)

        Image.fromarray(np.uint8(modified_image)).save(output_image_path)

        return modified_coeffs

    def extract_process(stego_image):
        
        image_array = load_image(stego_image)
        coeffs = perform_dwt(image_array)

        extracted_message = extract_message(coeffs)
        print("Extracted message:", extracted_message)

        # psnr_value = calculate_psnr(image_array, coeffs)
        # print("PSNR:", psnr_value)

        # capacity_value = calculate_capacity(coeffs)
        # print("Max Capacity:", capacity_value, "bytes")

        # bpp_value = calculate_bpp(extracted_message, image_array)
        # print("BPP:", bpp_value)
        
        # show_subbands(coeffs)


    embed_process(original_image_path, output_image_path, message_file)

    # Perform extraction independently
    extract_process(output_image_path)


if __name__ == "__main__":
    do_separate()
    


    


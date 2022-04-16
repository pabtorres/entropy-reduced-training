from reducciones_de_entropia_location_crop import *
from torchvision import transforms

def make_transformation(alpha):
    return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), Slice({alpha}),])')

def make_transformation_slice(alpha):
    return eval(f'transforms.Compose([Slice({alpha}),])')

def make_transformation_downsampling(alpha):
    return eval(f'transforms.Compose([Downsampling({alpha}),])')

def make_transformation_quantization(alpha):
    return eval(f'transforms.Compose([Quantization({alpha}),])')

def make_transformation_slice_selection(alpha, selection):
    if selection == 0:
        return eval(f'transforms.Compose([Slice({alpha})])')
    elif selection == 1:
        return eval(f'transforms.Compose([SliceTop({alpha})])')
    elif selection == 2:
        return eval(f'transforms.Compose([SliceBottom({alpha})])')
    elif selection == 3:
        return eval(f'transforms.Compose([SliceLeft({alpha})])')
    elif selection == 4:
        return eval(f'transforms.Compose([SliceRight({alpha})])')
    elif selection == 5:
        return eval(f'transforms.Compose([SliceUpRight({alpha})])')
    elif selection == 6:
        return eval(f'transforms.Compose([SliceUpLeft({alpha})])')
    elif selection == 7:
        return eval(f'transforms.Compose([SliceDownRight({alpha})])')
    elif selection == 8:
        return eval(f'transforms.Compose([SliceDownLeft({alpha})])')

def make_transformation_combined_selection(alpha, selection):
    if selection == 0:
        return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), Slice({alpha})])')
    elif selection == 1:
        return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), SliceTop({alpha})])')
    elif selection == 2:
        return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), SliceBottom({alpha})])')
    elif selection == 3:
        return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), SliceLeft({alpha})])')
    elif selection == 4:
        return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), SliceRight({alpha})])')
    elif selection == 5:
        return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), SliceUpRight({alpha})])')
    elif selection == 6:
        return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), SliceUpLeft({alpha})])')
    elif selection == 7:
        return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), SliceDownRight({alpha})])')
    elif selection == 8:
        return eval(f'transforms.Compose([Downsampling({alpha}), Quantization({alpha}), SliceDownLeft({alpha})])')

def make_transformations_for_slice(alpha):
    return [make_transformation_slice_selection(alpha, selection) for selection in range(9)]

def make_transformations_for_combined(alpha):
    return [make_transformation_combined_selection(alpha, selection) for selection in range(9)]

def make_transformation_selection(alpha, selection):
    if selection=='Combined': # Retorna una lista
        return make_transformations_for_combined(alpha)
    elif selection=='Crop': # Retorna una lista
        return make_transformations_for_slice(alpha)
    elif selection=='Resolution': # Retorna una sóla transformación
        return make_transformation_downsampling(alpha)
    elif selection=='Quantization': # Retorna una sóla transformación
        return make_transformation_quantization(alpha)
    else:
        raise Exception("Reducción no válida")
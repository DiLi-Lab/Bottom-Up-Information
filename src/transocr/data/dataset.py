from torch.utils.data import Dataset
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import re
import random

temp = True

# font for creating half-masked characters
# font_path = "/usr/share/fonts-droid-fallback/truetype/DroidSansFallbackFull.ttf"
# font = ImageFont.truetype(font_path, font_size)

font_path = '/swdata/yin/Cui/Re-Veil/NotoSansCJK-VF.ttf.ttc'
font_size = 40
font = ImageFont.truetype(font_path, font_size, index=2)
font.set_variation_by_axes([400.0]) # Set weight


# font_path = '/swdata/yin/Cui/Re-Veil/NotoSerifSC-VariableFont_wght.ttf'
# font_size = 40
# font = ImageFont.truetype(font_path, font_size)
# font.set_variation_by_axes([400.0]) # Set weight 



import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class trdgDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            label_file (str): Path to the text file with "filename label" on each line.
            image_dir (str): Directory containing the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.label_file = root
        self.image_dir = os.path.dirname(root)
        self.image_dir = os.path.join(self.image_dir, 'images')
        self.transform = transform

        # Load label file
        self.samples = pd.read_csv(self.label_file, sep=',')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_path'])
        label = row['label']
        label += '$'  # Append end-of-sequence character

        # Load image
        img = Image.open(img_path)  # Convert to grayscale if needed

        if self.transform:
            img = self.transform(img)
        
        return img, label, -1  # Return image, label, and size





class motrDataset(Dataset):
    def __init__(self, root=None, transform=None, revealed_part=None, reverse=False, extract_single_char=False):
        self.transform = transform
        self.revealed_part = revealed_part
        self.reverse = reverse
        self.extract_single_char = extract_single_char

        df = pd.read_csv(root, sep='\t')
        text_column='text'
    
        self.sentences = []
        split_pattern = re.compile(r'[^，。！？；：、“”‘’（）【】《》〈〉…—\-——@#\s]+|[，。！？；：、“”‘’（）【】《》〈〉…—\-——@#]')
        chinese_only = re.compile(r'[\u4e00-\u9fff]+')
        max_length = 1 if extract_single_char else 15

        for text in df[text_column].dropna():
            segments = re.findall(split_pattern, text)
            sentence = ''
            for token in segments:
                sentence += token
                if token in '，。！？；：、“”‘’（）【】《》〈〉…—-—— @#': #'，。！？；：@#()':
                    filtered = ''.join(re.findall(chinese_only, sentence))
                    if filtered:
                        # self.sentences.append(filtered)
                        self._add_sentence(filtered.strip(), max_length)
                    sentence = ''
            if sentence.strip():
                filtered = ''.join(re.findall(chinese_only, sentence))
                if filtered:
                    self.sentences.append(filtered)

    def _add_sentence(self, sentence, max_length):
        # Split sentence into chunks of max_length if it's too long
        if len(sentence) <= max_length:
            self.sentences.append(sentence)
        else:
            for i in range(0, len(sentence), max_length):
                chunk = sentence[i:i+max_length]
                self.sentences.append(chunk)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        label = self.sentences[idx]
        label += '$'
        img = generate_img_from_label(label[:-1], self.revealed_part)
        width, height = img.size
        if self.transform is not None:
            img = self.transform(img)
        return (img, label, (width, height))


class lmdbDataset_original(Dataset):
    def __init__(self, root=None, transform=None, img_from_lbl=False, revealed_part=None, reverse=False):
        """
        Params:
            img_from_lbl: whether to generate image with specified font from label
            revealed_part: part of the character to be revealed ('upper' or 'lower')
        """
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.img_from_lbl = img_from_lbl
        self.revealed_part = revealed_part
        self.reverse = reverse

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'index range error index: %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = strQ2B(label)
            label += '$'
            # label = label.lower()

            if self.img_from_lbl:
                img = generate_img_from_label(label[:-1], self.revealed_part)
            else:
                img_key = 'image-%09d' % index
                imgbuf = txn.get(img_key.encode())
                if imgbuf is None:
                    img_key = 'image_hr-%09d' % index
                    imgbuf = txn.get(img_key.encode())
                    if imgbuf is None:
                        raise ValueError('Corrupted image for %d' % index)

                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img = Image.open(buf).convert('RGB')
                    pass
                except IOError:
                    print('Corrupted image for %d' % index)
                    return self[index + 1]
                # Yin, 14 Mar 2025
                if self.revealed_part=='upper':
                    img = img.crop((0, 0, width, height // 2)) 
                if self.revealed_part=='lower':
                    img = img.crop((0, height // 2, width, height))

            width, height = img.size
            if self.transform is not None:
                img = self.transform(img)
        return (img, label, (width, height))


class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, img_from_lbl=False, revealed_part='full', part_labeled_img=False, reverse=False):
        """
        Params:
            img_from_lbl: whether to generate image with specified font from label
            revealed_part: part of the character to be revealed ('upper' or 'lower' or 'full' or 'random')
            part_labeled_img: whether to use the image together with the revealed-part label (3-valued) as input for the model
        """
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.img_from_lbl = img_from_lbl
        self.revealed_part = revealed_part
        self.reverse = reverse
        self.part_labeled_img = part_labeled_img

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index >= len(self):
            index = len(self) - 1
        assert index < len(self), 'index range error index: %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = strQ2B(label)
            if self.img_from_lbl:
                label = ''.join([char for char in label if is_chinese_char(char)])
                if not label:
                    return self[((index) % self.nSamples)]  # go to the next sample safely
            label += '$'
            # label = label.lower()

            if self.revealed_part == 'random':
                revealed_part = random.choice(['upper', 'lower', 'full'])
            else:
                revealed_part = self.revealed_part

            if self.img_from_lbl:
                img = generate_img_from_label(label[:-1], revealed_part)
            else:
                img_key = 'image-%09d' % index
                imgbuf = txn.get(img_key.encode())
                if imgbuf is None:
                    img_key = 'image_hr-%09d' % index
                    imgbuf = txn.get(img_key.encode())
                    if imgbuf is None:
                        raise ValueError('Corrupted image for %d' % index)

                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img = Image.open(buf).convert('RGB')
                    pass
                except IOError:
                    print('Corrupted image for %d' % index)
                    return self[index + 1]
                    
                if revealed_part=='upper':
                    img = img.crop((0, 0, width, height // 2)) 
                if revealed_part=='lower':
                    img = img.crop((0, height // 2, width, height))

            width, height = img.size
            if self.transform is not None:
                img = self.transform(img)

        if self.part_labeled_img:
            part_label = {"full": 0, "upper": 1, "lower": 2}[revealed_part]
            return (img, part_label, label, (width, height))
        else:
            return (img, label, (width, height))


def is_chinese_char(char):
    """Check if a character is a Chinese ideograph (excluding punctuation)."""
    return any([
        '\u4e00' <= char <= '\u9fff',         # CJK Unified Ideographs
        '\u3400' <= char <= '\u4dbf',         # Extension A
        '\U00020000' <= char <= '\U0002A6DF', # Extension B
        '\U0002A700' <= char <= '\U0002B73F', # Extension C
        '\U0002B740' <= char <= '\U0002B81F', # Extension D
        '\U0002B820' <= char <= '\U0002CEAF', # Extension E
        '\U0002CEB0' <= char <= '\U0002EBEF', # Extension F
    ])

# def is_chinese_char(char):
#     """Check if a character is a CJK (Chinese) character."""
#     return any([
#         '\u4e00' <= char <= '\u9fff',        # CJK Unified Ideographs
#         '\u3400' <= char <= '\u4dbf',        # CJK Unified Ideographs Extension A
#         '\u20000' <= char <= '\u2a6df',      # CJK Unified Ideographs Extension B
#         '\u2a700' <= char <= '\u2b73f',      # Extension C
#         '\u2b740' <= char <= '\u2b81f',      # Extension D
#         '\u2b820' <= char <= '\u2ceaf',      # Extension E
#         '\u2ceb0' <= char <= '\u2ebef',      # Extension F
#     ])

class lmdbDataset_singleCharExtracted(Dataset):
    def __init__(self, root=None, transform=None, img_from_lbl=False, revealed_part='full', part_labeled_img=False, reverse=False):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        self.transform = transform
        self.img_from_lbl = img_from_lbl
        self.revealed_part = revealed_part
        self.reverse = reverse
        self.part_labeled_img = part_labeled_img

        # Build index of (sample_idx, char_idx) for Chinese characters only
        self.index_map = []
        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode()))
            for i in range(1, self.nSamples + 1):
                label_key = 'label-%09d' % i
                label = txn.get(label_key.encode()).decode('utf-8')
                label = strQ2B(label).strip() + '$'
                for j, char in enumerate(label[:-1]):
                    if is_chinese_char(char):
                        self.index_map.append((i, j))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        sample_idx, char_idx = self.index_map[index]
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d' % sample_idx
            label = txn.get(label_key.encode()).decode('utf-8')
            label = strQ2B(label).strip() + '$'

            char = label[char_idx]

            if self.img_from_lbl:
                img = generate_img_from_label(char, self.revealed_part)
            else:
                img_key = 'image-%09d' % sample_idx
                imgbuf = txn.get(img_key.encode())
                if imgbuf is None:
                    img_key = 'image_hr-%09d' % sample_idx
                    imgbuf = txn.get(img_key.encode())
                    if imgbuf is None:
                        raise ValueError(f'Corrupted image for {sample_idx}')

                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img = Image.open(buf).convert('RGB')
                except IOError:
                    print('Corrupted image for %d' % sample_idx)
                    return self[index + 1]

                if self.revealed_part == 'upper':
                    img = img.crop((0, 0, width, height // 2))
                elif self.revealed_part == 'lower':
                    img = img.crop((0, height // 2, width, height))

            width, height = img.size
            if self.transform is not None:
                img = self.transform(img)

        # return img, char+'$', (width, height)
        if self.part_labeled_img:
            part_label = {"full": 0, "upper": 1, "lower": 2}[self.revealed_part]
            return (img, part_label, char+'$', (width, height))
        else:
            return img, char+'$', (width, height)





def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288: 
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

def generate_img_from_label(label, revealed_part):
    """
    Generate an image from the label with a specified part revealed."
    Parameters:
        - label: The text to be rendered.
        - revealed_part: The part of the character to be revealed ('upper' or 'lower'). If None, the whole character is revealed.
    Returns:
        - image: The generated image with the specified part revealed.
    """

    # Dummy image to calculate bounding box
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), label, font=font)  # (x0, y0, x1, y1)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Define canvas size with padding
    padding = 5
    canvas_width = text_width + padding * 2
    canvas_height = text_height + padding * 2

    # Create image
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)
    x = (canvas_width - text_width) // 2 - bbox[0]
    y = (canvas_height - text_height) // 2 - bbox[1]
    draw.text((x, y), label, fill="black", font=font)

    if revealed_part == 'upper':
        draw.rectangle([0, canvas_height // 2, canvas_width, canvas_height], fill="white")
    elif revealed_part == 'lower':
        draw.rectangle([0, 0, canvas_width, canvas_height // 2], fill="white")

    return image

   

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


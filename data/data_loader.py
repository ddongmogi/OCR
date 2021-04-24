import PIL.Image as Image
import numpy as np
import torch
import os
import random
from trdg.generators import GeneratorFromStrings
from jamo import h2j, j2hcj
import time

from utils import AttnLabelConverter

class DataLoader(object):
    def __init__(self,conf):
        if conf['phoneme_type']:
            from data.target_index import phoneme_index
            self.tar2ind,self.ind2tar = phoneme_index(paragraph_form=conf['paragraph_type'])
        else:
            from data.target_index import character_index
            self.tar2ind,self.ind2tar = character_index(conf['char_path'],paragraph_form=conf['paragraph_type'])
        
        self.phoneme_type = conf['phoneme_type']

        self.word_bag = self.make_word_bag(conf['word_path'])
        self.back_ground = self.make_back_ground(conf['back_path'])
        self.font_list = self.make_font_list(conf['font_path'])
        
        self.converter = AttnLabelConverter(self.tar2ind)
        self.paragraph_type = conf['paragraph_type']
        
    def make_font_list(self,font_path):
        return [os.path.join(font_path,i) for i in os.listdir(font_path)]
        
    def make_back_ground(self,back_path):
        return Image.open(back_path)
        
    def make_word_bag(self,word_path):
        with open(word_path,'r') as f:
            word_bag = f.readlines()
        
        #Delete '\n'
        for i,word in enumerate(word_bag):
            word_bag[i] = word.replace('\n','')
        
        #Delete unused character in word_bag, only when character mode
        if not self.phoneme_type:
            new_word_bag = []
            for i,word in enumerate(word_bag):
                word_triggerd = False
                for j in word:
                    if not j in self.tar2ind.keys():
                        word_triggerd = True
                        break
                if word_triggerd:
                    continue
                else:
                    new_word_bag.append(word)
            word_bag = new_word_bag
            
        return word_bag        
        
    def line_generator(self, row_max):
        row = None
        text_row = []
        x=0
        size = np.random.randint(32,56)
        font = [random.choice(self.font_list)]
        while True:
            generator = GeneratorFromStrings(
                [random.choice(self.word_bag)],
                count=1, #row_max define number of words in a text line
                blur=0,
                size=size,
                language='ko',
                #fonts=[random.choice(font_list) for _ in range(count)],
                fonts=font,
                random_blur=False,
                margins=(5,10,5,5),
            )

            for i,j in generator:
                img = i
                lbl = j
            if not row:
                #newpal = Image.new('RGB', (img.width,img.height))
                newrow = self.back_ground.crop((0,0,img.width,img.height))

                newrow.paste(img,(0,0))
                row = newrow
                x += img.width

                text_row.append(lbl)
            else:
                # row_max-(x+img.width)<=size*2 if remained space in row Image obj not enough to attach new word, return current obj and lbl
                if row_max-(x+img.width)>=0 and row_max-(x+img.width)<=size*2:
                    newrow = self.back_ground.crop((0,0,x+img.width,img.height))

                    newrow.paste(img,(x,0))
                    newrow.paste(row,(0,0))
                    row = newrow
                    x += img.width

                    text_row.extend(['\t',lbl])
                    break
                elif row_max-(x+img.width)<=0:
                    continue
                #newpal = Image.new('RGB', (x+img.width,img.height))
                newrow = self.back_ground.crop((0,0,x+img.width,img.height))

                newrow.paste(img,(x,0))
                newrow.paste(row,(0,0))
                row = newrow
                x += img.width
                text_row.extend(['\t',lbl])
                
        return row, text_row
    
    def paragraph_generator(self, row_max ,col_count,indent):
        palette = None
        paragraph = []
        y=0

        #Generate a paragraph
        for i in range(col_count):
            row, text_row = self.line_generator(row_max)

            #attach row image obj to palette image obj
            if not palette:
                palette = self.back_ground.crop((0,0,row.width + indent,row.height))
                palette.paste(row,(0 + indent,0))
                y += row.height

                paragraph.extend(text_row)
            else:
                if palette.width < row.width:
                    max_x = row.width
                else:
                    max_x = palette.width
                newpal = self.back_ground.crop((0,0,max_x,y+row.height))
                newpal.paste(palette,(0,0))
                newpal.paste(row,(0,y))
                palette = newpal
                y += row.height

                paragraph.append('\n')
                paragraph.extend(text_row)
        
        return palette, paragraph
    
    def batch_generator(self, batch_size, row_max ,col_count=5,indent=30):
        text_batch = []
        img_batch = []

        #per batch
        for ith in range(batch_size):
            palette = None
            paragraph = []
            y=0

            if self.paragraph_type:
                #Generate a paragraph
                palette, paragraph = self.paragraph_generator(row_max ,col_count,indent)

                #when successfully generated, add to batch list
                paragraph_jamo = []
                for i in paragraph:
                    for j in i:
                        paragraph_jamo.extend(j)
                text_batch.append(paragraph_jamo)
                img_batch.append(palette)
            else:
                #Generate row
                row, text_row = self.line_generator(row_max)
                
                #add to batch
                line_jamo = []
                for i in text_row:
                    for j in i:
                        line_jamo.extend(j)
                
                text_batch.append(line_jamo)
                img_batch.append(row)
        
        # if phoneme_type is true, convert to phoneme type text
        if self.phoneme_type:
            new_text = []
            for i in text_batch:
                text_jamo = []
                for j in i:
                    text_jamo.extend(list(h2j(j)))
                new_text.append(text_jamo)
            text_batch = new_text

        #padding each image in the batch as maximum size
        batch_width = max([i.width for i in img_batch])
        batch_height = max([i.height for i in img_batch])
        for i in range(batch_size):
            batpal = self.back_ground.crop((0,0,batch_width,batch_height))
            batpal.paste(img_batch[i],(0,0))
            img_batch[i]=batpal

        img_temp = [np.asarray(k)/255 for k in img_batch]
        img_temp = [np.transpose(k.astype(np.float32),(2,0,1)) for k in img_temp]
        img_temp = [torch.tensor(k[np.newaxis,:,:,:]) for k in img_temp]

        img_batch = torch.zeros(len(img_temp),3,batch_height,batch_width)

        #mono image mode
        #img_batch = torch.zeros(len(img_temp),1,batch_height,batch_width)
        for i,data in enumerate(img_temp):
            img_batch[i,:,:data.shape[-2],:data.shape[-1]] = data[:,:-1,:,:]

            #mono image mode
            #img_batch[i,:,:data.shape[-2],:data.shape[-1]] = data[:,0,:,:]
            
        #pad paragraph
        batch_max_length = max([len(i) for i in text_batch])
        text_batch = self.converter.encode(text_batch,batch_max_length)
        
        return img_batch, text_batch



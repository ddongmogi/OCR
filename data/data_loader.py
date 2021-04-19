import PIL.Image as Image
import numpy as np
import torch
import os

class DataLoader(object):
    def __init__(self,conf):
        if conf['phoneme_type']:
            from data.target_index import phoneme_index
            self.tar2ind,self.ind2tar = phoneme_index()
        else:
            from data.target_index import character_index
            self.tar2ind,self.ind2tar = character_index(conf['char_path'])
            
        self.word_bag = self.make_word_bag(conf['word_path'],self.tar2ind,conf['phoneme_type'])
        self.back_ground = self.make_back_ground(conf['back_path'])
        self.font_list = self.make_font_list(conf['font_path'])
        
    def make_font_list(self,font_path):
        return [os.path.join(font_path,i) for i in os.listdir(font_path)]
        
    def make_back_ground(self,back_path):
        return Image.open(back_path)
        
    def make_word_bag(self,word_path,tar2ind,phoneme=False):
        with open(word_path,'r') as f:
            word_bag = f.readlines()
        
        #Delete unused character in word_bag, only when character mode
        if not phoneme:
            while True:
                except_bag = []
                for i,word in enumerate(word_bag):
                    for char in word:
                        try:
                            tar2ind[char]
                        except:
                            except_bag.append(char)
                            del word_bag[i]
                except_bag = set(except_bag)

                if len(except_bag)<1:
                    break
                
        return word_bag        
        
        
        
    def batch_generator(self, batch_size, row_max ,col_count,indent=30):

        para_batch = []
        img_batch = []

        #per batch
        for ith in range(batch_size):

            palette = None
            paragraph = []
            y=0

            #Generate a paragraph
            for i in range(col_count):
                row = None
                text_row = []
                x=0
                size = np.random.randint(32,56)
                font = [random.choice(font_list)]
                while True:
                    generator = GeneratorFromStrings(
                        [random.choice(word_bag).replace('\n','') for _ in range(count)],
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
                        newrow = back.crop((0,0,img.width,img.height))

                        newrow.paste(img,(0,0))
                        row = newrow
                        x += img.width

                        text_row.append(lbl)
                    else:
                        if row_max-(x+img.width)>=0 and row_max-(x+img.width)<=size*2:
                            newrow = back.crop((0,0,x+img.width,img.height))

                            newrow.paste(img,(x,0))
                            newrow.paste(row,(0,0))
                            row = newrow
                            x += img.width

                            text_row.extend(['\t',lbl])
                            break
                        elif row_max-(x+img.width)<=0:
                            continue
                        #newpal = Image.new('RGB', (x+img.width,img.height))
                        newrow = back.crop((0,0,x+img.width,img.height))

                        newrow.paste(img,(x,0))
                        newrow.paste(row,(0,0))
                        row = newrow
                        x += img.width

                        text_row.extend(['\t',lbl])


                #attach row image obj to palette image obj
                if not palette:
                    palette = back.crop((0,0,row.width + indent,row.height))
                    palette.paste(row,(0 + indent,0))
                    y += row.height

                    paragraph.extend(text_row)
                else:
                    if palette.width < row.width:
                        max_x = row.width
                    else:
                        max_x = palette.width
                    newpal = back.crop((0,0,max_x,y+row.height))
                    newpal.paste(palette,(0,0))
                    newpal.paste(row,(0,y))
                    palette = newpal
                    y += row.height

                    paragraph.append('\n')
                    paragraph.extend(text_row)

            #when successfully generated, add to batch list
            paragraph_jamo = []
            for i in paragraph:
                for j in i:
                    paragraph_jamo.extend(j)
            para_batch.append(paragraph_jamo)
            img_batch.append(palette)

        #padding each image in the batch as maximum size
        batch_width = max([i.width for i in img_batch])
        batch_height = max([i.height for i in img_batch])
        for i in range(batch_size):
            batpal = back.crop((0,0,batch_width,batch_height))
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
        batch_max_length = max([len(i) for i in para_batch])
        para_batch = converter.encode(para_batch,batch_max_length)

        return img_batch, para_batch



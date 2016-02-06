require 'nn'
require 'optim'
require 'torch'
require 'nn'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
cjson=require('cjson') 
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_json','data_prepro.json','path to the json file containing vocab and answers')
cmd:option('-image_root','','path to the image root')
cmd:option('-cnn_proto', '', 'path to the cnn prototxt')
cmd:option('-cnn_model', '', 'path to the cnn model')
cmd:option('-batch_size', 10, 'batch_size')

cmd:option('-out_name', 'data_img.h5', 'output name')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)
net=loadcaffe.load(opt.cnn_proto, opt.cnn_model,opt.backend);
net:evaluate()
net=net:cuda()

function loadim(imname)
    im=image.load(imname)
    im=image.scale(im,224,224)
    if im:size(1)==1 then
        im2=torch.cat(im,im,1)
        im2=torch.cat(im2,im,1)
        im=im2
    elseif im:size(1)==4 then
        im=im[{{1,3},{},{}}]
    end
    im=im*255;
    im2=im:clone()
    im2[{{3},{},{}}]=im[{{1},{},{}}]-123.68
    im2[{{2},{},{}}]=im[{{2},{},{}}]-116.779
    im2[{{1},{},{}}]=im[{{3},{},{}}]-103.939
    return im2
end

local image_root = opt.image_root
-- open the mdf5 file

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
    table.insert(train_list, image_root .. imname)
end

local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
    table.insert(test_list, image_root .. imname)
end

local ndims=4096
local batch_size = opt.batch_size
local sz=#train_list
local feat_train=torch.CudaTensor(sz,ndims)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    ims=torch.CudaTensor(r-i+1,3,224,224)
    for j=1,r-i+1 do
        ims[j]=loadim(train_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_train[{{i,r},{}}]=net.modules[43].output:clone()
    collectgarbage()
end

print('DataLoader loading h5 file: ', 'data_train')
local sz=#test_list
local feat_test=torch.CudaTensor(sz,ndims)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i, sz)    
    r=math.min(sz,i+batch_size-1)
    ims=torch.CudaTensor(r-i+1,3,224,224)
    for j=1,r-i+1 do
        ims[j]=loadim(test_list[i+j-1]):cuda()
    end
    net:forward(ims)
    feat_test[{{i,r},{}}]=net.modules[43].output:clone()
    collectgarbage()
end

local train_h5_file = hdf5.open(opt.out_name, 'w')
train_h5_file:write('/images_train', feat_train:float())
train_h5_file:write('/images_test', feat_test:float())
train_h5_file:close()


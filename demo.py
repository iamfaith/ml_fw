# torch.save(model.state_dict(), 'train_valid_exp4-epoch{}.pth'.format(epoch)) 


# if os.path.exists(checkpoint_file):
#     if config.resume:
#         checkpoint = torch.load(checkpoint_file)
#         model.load_state_dict(checkpoint['model'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
        

# def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
#     # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
#     start_epoch = 0
#     if os.path.isfile(filename):
#         print("=> loading checkpoint '{}'".format(filename))
#         checkpoint = torch.load(filename)
#         start_epoch = checkpoint['epoch']
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         losslogger = checkpoint['losslogger']
#         print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(filename, checkpoint['epoch']))
#     else:
#         print("=> no checkpoint found at '{}'".format(filename))

#     return model, optimizer, start_epoch, losslogger


# torch.cuda.empty_cache()

from common import Decorator

@Decorator.wandb
def hellow():
    print('aaa')
    
    
if __name__ == "__main__":
    hellow()
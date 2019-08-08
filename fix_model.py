import torch
import copy
import sys, getopt

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('fix_model.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print('Input file is "', inputfile)
   print('Output file is "', outputfile)

   ckpt_file = inputfile
   model = torch.load(ckpt_file)
   KEYS_1 = model['state_dict'].keys()
   KEYS_list = [each for each in KEYS_1]
   print('initial number of keys = ', len(KEYS_list))
   print(KEYS_list)
   for each in KEYS_list:
       if 'float_float' in each:
          print(each)
          del model['state_dict'][each]
          print('removing key:', each)
   KEYS = model['state_dict'].keys()
   print('final number of keys = ', len(KEYS))
   torch.save(model, outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])

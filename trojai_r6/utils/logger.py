import logging,os
from colorama import Fore,Style 

 
class Logger:
 def __init__(self, path,clevel = logging.DEBUG,Flevel = logging.DEBUG):
  self.logger = logging.getLogger(path)
  self.logger.setLevel(logging.DEBUG)
  fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
  sh = logging.StreamHandler()
  sh.setFormatter(fmt)
  sh.setLevel(clevel)
  fh = logging.FileHandler(path)
  fh.setFormatter(fmt)
  fh.setLevel(Flevel)
  self.logger.addHandler(sh)
  self.logger.addHandler(fh)



 def trigger_generation(self,message):
  self.logger.info(Fore.GREEN + '[Trigger Generation]: ' + message + Style.RESET_ALL)

 def result_collection(self,message):
  self.logger.info(Fore.YELLOW + '[Scanning Result]: ' + message + Style.RESET_ALL)

 def best_result(self,message):
  self.logger.info(Fore.RED + '[Best Estimation]: ' + message + Style.RESET_ALL)


 def debug(self,message):
  self.logger.debug(message)
 
 def info(self,message):
  self.logger.info(Fore.YELLOW + message + Style.RESET_ALL)
 
 def war(self,message):
  self.logger.warning(message)
 
 def error(self,message):
  self.logger.error(message)
 
 def cri(self,message):
  self.logger.critical(message)

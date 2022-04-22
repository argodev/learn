# Chapter 01

- Setup Python Environment. Author recommends using Kali. I'm going to try to 
  stick with my existing Python environ.
- I made a virtual enviornment (`blackhat_python`) that will be set for this 
  book
- validated that I could install `github3.py` and also setup a requirements.txt 
  file in the root of this book's directory
- I skipped the section on installing WingIDE
- I did figure that it would be good to know how to debug my python code using a 
  real debugger (vs. just `print()` commands), so I wrote the script the author 
  suggested (`number_converter.py`) and tested it (it worked).
- using `pdb` (the python debugger) looks pretty straight forward:

```bash
$ python -m pdb number_converter.py
(Pdb) l (list source code)
(Pdb) b 17
(Pdb) n (next)
(Pdb) c (continue)
(Pdb) s (step into)
(Pdb) print(var_name)
```


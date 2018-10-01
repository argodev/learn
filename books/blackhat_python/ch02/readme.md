you can run the netcat replacment in 'server' mode as follows:

```bash
$ python bhpnet.py -l -p 9999 -c
```

The client mode can then be run like so:

```bash
$ python bhpnet.py -t localhost -p 9999
```

Another example is as follows:

```bash
echo -ne "GET / HTTP/1.1\r\nHost: www.google.com\r\n\r\n" | python bhpnet.py -t www.google.com -p 80
```




# Chapter 2

This is where we get to really start doing things. I will admit that it took me a while to really roll on this one as I spent way too much time getting my environment setup. I am currently trying the WSL2 thing and vscode remote. As such, I resisted the urge to install a VM platform on my laptop and, instead, configured and tested remote access to my always-on desktop. VSCode Remote seems reasonably happy and I am making much better progress. We'll see how well this works moving forward.

## TCP Port Scanner

### Testing Ports for Availability

This example (`/01-dial`) was pretty straight forward, but (see above) too awhile as I wanted to get a host that I could scan without triggering anyone's alarms. Also, I with the "I'm a Moron" award as I couldn't figure out why I was getting a failure when attempting to connect to machine:21 until I realized that SSH listens on port _22_ rather than _21_. It's amazing how much better things worked after that.

### Slow Scanner

This solution (`/02-tcp-scanner-slow`) was nice and simple. Easy to understand and, at least with a local machine to scan, really wasn't slow at all. That said, it is obvious that they are referring to the non-concurrency of it all and that leads us to the next stage

### Too-Fast Scanner

As suggested, my time resulst showed similar to the text and it finished nearly immediately.

```bash
❯ time ./main
./main  0.01s user 0.02s system 60% cpu 0.051 total
```

### WaitGroup Approach

This approach (`/04-tcp-scanner-wg-too-fast`) seemed to work for me, although I'll trust the authors in that it isn't ideal. 

### Port Scanning with a Worker Pool

This (`/05-tcp-sync-scanner`) is similar to the previous, but uses worker pools to ensure consistency. The first draft of this compiled and ran, producing a quite un-ordered list of numbers.

### Port Scanning with Multichannel Communication

This solution (`/06-tcp-scanner-final`) is the final step in the text for the port scanner and it is interesting as it ties together and demonstrates some interesting facets of GO's concurrency routines. On my system this continued to scan quite rapidly and also had the correct results (only one port open).

```bash
❯ time ./main
22 open
./main  0.06s user 0.07s system 372% cpu 0.036 total
```

## TCP Proxy


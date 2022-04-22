# CHAPTER 1

I didn't follow the specific installation instructions, esp. since I think they were outdated (setting `GOROOT`, `GOPATH`, etc.). Instead, I followed the installation instructions located at [https://golang.org/doc/install](https://golang.org/doc/install).

```bash
$ go version
go version go1.15.6 linux/amd64
```

I chose to use MS VSCode as my development tool. From within WSL, I simply created a `*.go` file, ran `$ code .` and then, when trying to edit the file, VS Code prompted me to install the appropriate extension and related linting tools. I accepted all of the defaults.

Running with the flags to strip the binary did, in my case, drop the size significantly. `hello` went from nearly 2 MB down to 1.3 MB. Still quiet large relative to the original _91 byte_ source file.

## Cross Complilation

Specifically noting this as I will need it in the future.

```bash
# complile for rpi
$ GOOS="linux" GOARCH="arm64" go build -o hello_arm hello.go
$ file hello_arm
hello_arm: ELF 64-bit LSB executable, ARM aarch64, version 1 (SYSV), statically linked, Go BuildID=K6W8fVeBDLNnCIaEVOfX/e8U3TY4Uj9S7bXfa1FHR/YlpMjKv0iduuQw9s2tfp/CGYxELcTOCW286zemR7F, not stripped
```


## Random helpful commands

```bash
$ go doc <cmd>

$ go get <pkg>
$ go get github.com/stacktitan/ldapauth
```

Also was reminded of the utility of the `tree` command, and the value of installing it

## Primitive Data Types

```go
var x = "Hello World"
z := int(42)
```

## Slices and maps

```go
var s = make([]string, 0)
var m = make(map[string]string)
s = append(s, "some string")
m["some key"] = "some value"
```

## Pointers, Structs, and Interfaces

```go
var count = int(42)
ptr := &count
fmt.Println(*ptr)
*ptr = 100
fmt.Println(count)
```

```go
type Person struct {
    Name string
    Age int
}

func (p *Person) SayHello() {
    fmt.Println("Hello, ", p.Name)
}

func main() {
    var guy = new(Person)
    guy.Name = "Dave"
    guy.SayHello()
}
```

Remember, __Capital__ letter-named items are _public_ and accessible outside the package. __lower case__ functions and and variables are private.

```go
type Friend interface {
    SayHello()
}
```

```go
func Greet(f Friend) {
    f.SayHello()
}

func main() { 
    var guy = new(Person)
    guy.Name = "Dave"
    Greet(guy)
}
```

## Control Structures

```go
if x == 1 {
    fmt.Println("X is equal to 1")
} else {
    fmt.Println("X is not equal to 1")
}
```

```go
switch x {
    case "foo":
        fmt.Println("Found foo")
    case "bar":
        fmt.Println("Found bar")
    default:
        fmt.Println("Default case")
}
```

Note, Type Switches can also be particularly helpful

```go
func foo(i interface{}) {
    switch v := i.(type) {
        case int:
            fmt.Println("I'm an integer!")
        case string:
            fmt.Println("I'm a string!")
        default:
            fmt.Println("Unknown Type!")
    }
}
```

```go
// normal mode
for i := 0; i < 10; i++ {
    fmt.Println(i)
}

// foreach style
nums := []int{2,4,6,8}
for idx, val := range nums {
    fmt.Println(idx, val)
}
```

## Concurrency

```go
func f() {
    fmt.Println("f function")
}

func main() {
    go f()
    time.Sleep(1 * time.Second)
    fmt.Println("main function")
}
```

We can use channels to provide synchronization and communication between multiple threads/instances of go routines

```go
func strlen(s string, c chan int) {
    c <- len(s)
}

func main() {
    c := make(chan int)
    go strlen("Salutations", c)
    go strlen("World", c)
    x, y := <-c, <-c
    fmt.Println(x, y, x+y)
}
```

More coming on this topic, including buffered channels, wait groups, and mutexes.

## Error Handling

There is no try/catch/finally construct.

There is a built-in error type, defined as follows

```go
type error interface {
    Error() string
}
```

This can be mapped-on to any data type that provides an `Error` function, as shown here:

```go
type MyError string

func (e MyError) Error() string {
    return string(e)
}
```

A common pattern used for errors is as follows

```go
func foo() error {
    return errors.new("Some error Occured")
}

func main() {
    if err := foo(); err != nil {
        // handle the error
    }
}
```

## Structured Data

```go
type Foo struct {
    Bar string
    Baz string
}

func main() {
    f := Foo{"Joe Junior", "Hello Shabado"}
    b, _ := json.Marshal(f)
    fmt.Println(string(b))
    json.Unmarshal(b, &f)
}
```

I don't understand all of it quite at this time, but the syntax for decorating fields to help the serialization routine is important. Here's an example from the text:

```go 
type Foo struct {
    Bar string `xml:"id,attr"`
    Baz string `xml:"parent>child"`
}
```
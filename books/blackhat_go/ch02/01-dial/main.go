package main

import (
	"fmt"
	"net"
)

func main() {
	_, err := net.Dial("tcp", "argox1.ornl.gov:21")
	if err == nil {
		fmt.Println("Connection successful")
	} else {
		fmt.Println(err.Error)
	}
}

package main

import (
	"fmt"
	"net"
)

func main() {
	_, err := net.Dial("tcp", "172.16.130.135:22")
	if err == nil {
		fmt.Println("Connection successful")
	} else {
		fmt.Println(err.Error)
	}
}

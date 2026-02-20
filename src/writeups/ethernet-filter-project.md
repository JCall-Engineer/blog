---
title: "Hardware Ethernet Filter (USU ECE 3710)"
description: "Archived documentation of a hardware Ethernet filter project from USU ECE 3710, preserving design details, implementation, and original project materials."
tags: [ projects, engineering ]
---

This project from December 2015 was originally hosted on USU Spaces, which has since been retired. This page simply preserves the original documentation and artifacts.

## Overview

This project was designed and completed by:

- John Call - johnkc1992@gmail.com
- Landon Wilcox - landon.wilcox@outlook.com

Feel free to email us with questions about our project.

---

![Project Breadboard](@assets/20151218_121301.jpg)

The applications of internet filtering are widespread; examples include: parental controls, ad-blocking, and cyber security. A hardware solution allows for one device to protect an entire network, where most existing software solutions will only protect one computer (and have significant limitations in regard to parental controls).

In our project, we filtered against only 1 website: bing.com. Over the course of the project, we experimented with different methods of filtering data. In the end we decided to filter based on if a packet's destination IP matched the known IP of bing's server.

For a demonstration, see [https://youtu.be/A96wC7NZhdE](https://youtu.be/A96wC7NZhdE)

Below are speed test results showing the ramifications of using our filter.

<style>
	table img {
		width: 300px;
	}
</style>

| Direct Connection Speed Test                                                    | Speed Test with Filter Disabled                                                     | Speed Test with Filter Enabled                                                    |
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| ![Direct Connection Speed Test](@assets/Speed%2520-%2520Direct%2520Connect.png) | ![Speed Test with Filter Disabled](@assets/Speed%2520-%2520Filter%2520Disabled.png) | ![Speed Test with Filter Enabled](@assets/Speed%2520-%2520Filter%2520Enabled.png) |

There is high variability in the speed of using our device. This is because packets are only transferred 1 at time, resulting in an internet speed that is bottle-necked by the speed of our SPI communication with the WIz550io (ethernet) chips. As such, considering many programs are running on my laptop that use the internet, a program being silent during a test and active in another can account for why the test proved faster with the filter disabled.

What should be noted is that the filtration process itself achieves O(1), causing no noticeable difference between having the filter enabled or disabled. Methods to increase the speed of SPI communication between the Wiz550io chips we use and the processing speed of the microcontroller would be the best course of study to increase the the bottle-necked speed closer to the direct connection.

### Required Parts

- 1 Tiva-C TM4C123G Launchpad
- 1 ER-TFTM032 LCD/Touchscreen Module
- 2 Wiz550io modules
- An external 3.3V source (sources up to 350mA)
- A router with internet access
- A laptop/desktop with an Ethernet port
- Two (non-crossover) Ethernet cables
- Wireshark (software) may prove useful

### External Resources

- [Wiz550io Home Page](http://www.wiznet.co.kr/product-item/wiz550io/)
- [Wiz550io Wiki](http://wizwiki.net/wiki/doku.php?id=products:wiz550io:allpages)
- [W5500 DataSheet](@assets/w5500_ds_v106e_141230.pdf)
- [Microcontroller DataSheet](@assets/tm4c123gh6pm.pdf)
- [Tiva Launchpad DataSheet](@assets/spmu296.pdf)
- [LCD](@assets/ILI9341_DS_V1.10_20110415.pdf), [Touchscreen](@assets/LCD-XPT2046-EN.pdf), and [TFT-Module](@assets/ER-TFTM032-3_Datasheet.pdf) Datasheets
- [Wireshark Download Page](https://www.wireshark.org/#download)

### Our Project Design Documents

- [Initial Proposal](@assets/Hardware%20Ethernet%20Filter%20-%20Final%20Project%20Proposal.pdf)
- [Design Document](@assets/Hardware%20Ethernet%20Filter%20-%20Design%20Document.pdf)
- [Source Code](https://github.com/JCall-Engineer/ECE3710) (include the following:)
  - Final Project
    - main.c
    - enet.c
    - filter.c
  - Shared
    - fonts.c
    - LCD.c
    - SPI.c

## Connecting Everything

To the right are the schematics of how our microcontroller is configured to interface with the 3 peripherals. Something to note when wiring: we noticed that electromagnetic interference was still a problem even with only two devices sharing the same SPI module if the clock wire was too close to the data wire. Length of wire, or other factors we were not fully able to understand may be influences in that regard. In any case, if you have issues, try distancing the clock wires from MOSI/MISO.

Note the distance between the yellow (clock) and pink (MOSI/MISO) wires below. Without this, SPI communication errors result. See the Final Report for more details.

![Spacing Visualization](@assets/spi%2520wiring.jpg)

![Schematic](@assets/Schematic%2520-%2520Revision%25202.png)

![Wiz550io Pinmap](@assets/pinmap_wiz550io.jpg)

## Obtaining IP Information

Part of the project requires hard coding ip configuration (found at the beginning of enet.c). To determine what information you should use, use ipconfig (Windows) or ifconfig (linux). Additionally to determine the physical address of the router you are normally attached to, wireshark may prove useful.

The screen captures below will walk someone who is unfamiliar with internet configuration through the process, on Windows, of finding your ip configuration, and changing your adapter to use a static ip (which this project needs to function as it currently is; future improvements would include handling dynamic ip resolution with DHCP packets). Also, DNS servers are a part of DHCP resolution so you will have to select your own. In the screen capture you can see I used googles DNS servers (8.8.8.8 and 8.8.4.4)

The last screen capture demonstrates how the physical address of the router you are connected to can be obtained from a wireshark capture searching for the ip address of the default gateway.

![Obtaining Static IP Step 1](@assets/0%2520-%2520Obtaining%2520IP%2520Info.png)
![Obtaining Static IP Step 2](@assets/1%2520-%2520Network%2520and%2520Sharing%2520Center.png)
![Obtaining Static IP Step 3](@assets/2%2520-%2520Adapter.png)
![Obtaining Static IP Step 4](@assets/3%2520-%2520Properties.png)
![Obtaining Static IP Step 5](@assets/4%2520-%2520IPv4%2520Config.png)
![Obtaining Static IP Step 6](@assets/5%2520-%2520Static%2520Config.png)
![Obtaining Static IP Step 7](@assets/using%2520wireshark.png)

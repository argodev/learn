---
title: CSC7575 - Final Exam
author: Rob Gillen
header-includes:
    - \usepackage{fancyhdr}
    - \pagestyle{fancy}
    - \fancyhead[RO,RE]{Final Exam}
    - \fancyhead[LO,LE]{Rob Gillen, T00215814}
    - \usepackage{tikz}
    - \usetikzlibrary{calc,shapes.multipart,chains,arrows}
---

# Question 1

_Is the PLC logger presented in the paper useful for volatile memory forensics? If yes, how? If not, why not?[1]_

The answer to this question depends on your perspective (in a few ways). As detailed in this paper, the authors state that it is _not_. They cite its failures to adhere completely to the referenced NIST standard as to the characteristics of a proper volatile memory forensics tool. If you have a requirement of strict adherence to this standard, then one cannot but agree with their conclusions.

However, having worked with a number of investigators over my career, this simply doesn't hold water. There _are_ some things that are non-negotiables (e.g. ensuring a write-blocker is used or the device prevents tampering). Beyond those items most are somewhat mercenary in their approach. They'll take anything that will give them more insight into what went on. Obviously there are some tools that have legal implications and therefore must provide certain functionality (e.g. adherence to the standards referenced above), but many will use whatever tool (so long as it is non-destructive_) that moves the investigation forward. Often, a non-court-approved tool will give them the hint they need to then go back to a tool-of-record and find (for official records) the same piece of information.

In most of the referenced failed test cases, the tool failed due to a failure to notify the user of a change in state (connected, disconnected, etc.). These are certainly nice features, but may not reach the level of _required_ for adding value. One of the failed scenarios points to an uncertainty in whether or not the tool altered the state of the device. Unfortunately, it "failed" not because of confirmed state manipulation, but due to the fact that the researchers were unable to determine either way. While I agree that conclusive evidence is needed, I am uncertain that this should black-ball the tool. Further, if you insist on using a network-connected tool for harvesting this information (thus interacting with the device's local network stack) you may, by definition, be elminiating your tool from consideration due to enviornment manipulation.

The paper did show the device's ability to successfully retrieve the memory and to provide the investigators with enough information to determine what was going on, and what the attackers were doing (supporting hypothesis #1). This is a clear "adding of value" and demonstration of at least some usefulness.

In the final anlysis, the question should be rephrased or dissected into a few. Rather than the authors asking is the PLC logger "useful" (a broad, and somewhat nebulous term), they might ask one or more of the following:

1. Does the PLC logger adhere to the NIST standard?
1. Does the PLC logger provide information that contributes to a forensic investigation?
1. What attributes would disqualify a tool (e.g. the PLC logger) from being used in any fashion in a forensic investigation?
1. At what stage, or in what capacity, might the PLC logger be used in a forensic investigation?

# Question 2

_What is “betweenness” in a graph? How might unexpectedly disconnecting a link with high betweenness effect the stability of the electric grid?[2]_

"Betweeness" is often more completely referred to as "Betweeness Centrality" or sometimes just as "Centrality". While referenced in this paper, it a general graph-theory term/definition and is not unique to these authors. The concepted of betweeness centrality is mostly easily understood by executing the following steps:

1. For every pair of verticies in a graph, calculate the shortest path connecting them
1. For each vertex that is traversed by one of these shortest paths, increment a vertex-specific counter totalling how many times it has been crossed
1. Sort the verticies by the vertex-specific counter... this counter is a measure of _betweeness_ or _centrality_

It is essentially a measure of how critical a given node is to the optimal (shortest paths) transversal of a graph.

As regards the stability of the electric grid... if a given node has a high betweeness value, that indicates that many other nodes depend on it as a primary route of their power source. If this node were to be taken out, then secondary routes/paths would be utlized. This could have the following effects:

1. The flow of electricity is now on non-optimal paths, potentially increasing the transmission loss and increasing the cost of delivering the service.
1. Like any fault-tolerant system, sometimes the secondary paths are not designed to support the same scale or load as the primary paths. The redirection of load onto secondary paths may inadvertantly overload particular paths, causing cascading failures which can ripple/continue until stability is achieved.

# Question 3

_SCADA network traffic is often considered to be consistent and reliable (in contrast to IT traffic). Do the authors consider this to be true? Why or why not? Do you agree? Why or why not? [3]_

The authors are unclear as to their initial position, however the hypothesis (there exists a need to confirm/debunk commonly held opinions as to the characteristics of said traffic) leads a reader to assume they are at least, to some degree, skeptical. Regardless, it is a clear and easy-to-rationalize position and has some basis in historical emperical testing. SCADA systems (historically) were closed networks that operated with fixed protocols often driven by clock-syncronized circuitry. In this case, the traffic is, due to clock cycle-locks, very regular. The traffic on the CANBUS in modern automobilies is another example of a closed-loop system that has very regular traffic patterns.

What the authors show based on their longitudinal study, is that _modern_ SCADA systems (e.g. DNP3 over IP) do not exibit this same stability or consistency, even in drastically over-provisioned networks. They show that poll time and inter-packet arrival times have wide distributions rather than the narrow variance one would expect. Similarly, they show that TCP flow durations and flow size are much shorter than would be anticipated in long-connected, lock-step communications.

My agreement (or lack thereof) is essentially a non-issue. Meaning, given a lack of my own emperical evidence, I do not have a position from which to argue. I do agree with the notion that the original assumption (inherent consistency of traffic) is a fair and assumed position by most (myself included). I would further state, however, that their results are intrguing and present opportunities to study other similar networks to estabilish the broad applicability of their results (e.g. ICS protocols over modern networks such as TCP/IP do not inherit their former regularlity).

# Question 4

_What do the authors identify as the largest source of system instability? Why?[3]_

This slide deck was delivered on July 2, 2018 and there have been no official publications since this point. That said, the dissertaion itself centers around four journal publications, one of which has yet to be published (it has, however, been accepted for publication). The three articles that are currently available are as follows:

- Sherrell R. Greene, “Are Current U.S. Nuclear Power Plants Grid Resilience Assets?” Nucl. Technol., 202, 1 (2018); [https://doi.org/10.1080/00295450.2018.1432966](https://doi.org/10.1080/00295450.2018.1432966)
- Sherrell R. Greene, “Nuclear Power: Black Sky Liability or Black Sky Asset?” Int. J. Nucl. Security, 2, 3 (2016), [http://dx.doi.org/10.7290/V78913SR](http://dx.doi.org/10.7290/V78913SR)
- Sherrell R. Greene, “The Key Attributes, Functional Requirements, and Design Features of Resilient Nuclear Power Plants (rNPPs),” Nucl. Technol. (accepted for publication); [https://doi.org/10.1080/00295450.2018.1480213](https://doi.org/10.1080/00295450.2018.1480213)

# Question 5

_Why is there a challenge to “black start” of nuclear power plants? How do the authors propose to address this challenge?[4]_

According to Dr. Greene, what is the primary reason that Nuclear Power Plants are strong candidates as anchors in the US grid resiliency program?

> __Fuel Security (at any given point in time, a Nuclear Plant has an average of 1 year's worth of fuel in reserves compared to 2-3 days for fossil-fuel based generation facilities.)__

- What is one of the challenges that prevents Nuclear Power Plants from being used as _"Islands"_ in the recovery process?

> __They have an inability to cold start (black start)__

- What is Dr. Greene's design recommendation (technical, not policy) for soliving the Black Start problem?

> __To design future nuclear power plants as a series of smaller modules such that each module was easy to run and one module could cold start another and the two could support the starting of successfully larger modules until the normal power output was once again achieved.__

## References

1. Wu, Tina, and Jason Nurse. “Exploring The Use Of PLC Debugging Tools For Digital Forensic Investigations On SCADA Systems.” Journal of Digital Forensics, Security and Law10, no. 4 (December 31, 2015): 79–96. http://ojs.jdfsl.org/index.php/jdfsl/article/view/347

2. Cai, Ye, Yong Li, Yijia Cao, Wenguo Li, and Xiangjun Zeng. “Modeling and Impact Analysis of Interdependent Characteristics on Cascading Failures in Smart Grids.” International Journal of Electrical Power & Energy Systems 89 (July 1, 2017): 106–14. https://doi.org/10.1016/j.ijepes.2017.01.010

3. Formby, David, Anwar Walid, and Raheem Beyah. “A Case Study in Power Substation Network Dynamics.” Proc. ACM Meas. Anal. Comput. Syst. 1, no. 1 (June 2017): 19:1–19:24. https://doi.org/10.1145/3084456

4. Greene, Sherrell R. “Are Current U.S. Nuclear Power Plants Grid Resilience Assets?” Nuclear Technology 202, no. 1 (April 3, 2018):  1–14. https://doi.org/10.1080/00295450.2018.1432966

# CMPE239StackOverFlow


# Problem Statement:

One advantage of large scale collaborative sites like Wikipedia and StackOverflow is the speed at which new information accumulates. Some questions on StackOverflow seemed to get answered immediately while others sit around for a while. As you are writing your question, it would be cool to have an idea of how long it might take to get answered. How well can you predict the time until a question gets answered? This will likely depend on the subject matter of the question, since some programming communities are more active than others.

# Dataset Attributes:
The matrix it contains has the following attributes:

qid: Unique question id
i: User id of questioner
qs: Score of the question
qt: Time of the question (in epoch time)
tags: a comma-separated list of the tags associated with the question. Examples of tags are ``html'', ``R'', ``mysql'', ``python'', and so on; often between two and six tags are used on each question.
qvc: Number of views of this question (at the time of the datadump)
qac: Number of answers for this question (at the time of the datadump)
aid: Unique answer id
j: User id of answerer
as: Score of the answer
at: Time of the answer

# Results:
![alt text](http://i66.tinypic.com/2n1gsnt.jpg)

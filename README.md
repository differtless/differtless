# Team #33 final project for CS107/AC207

![logo](https://github.com/differtless/cs107-FinalProject/blob/master/docs/img/differtless_logo.png)

## Team members

* Anastasia Ershova aershova@g.harvard.edu
* Mark Penrod mpenrod@g.harvard.edu
* Will Wang willwang@g.harvard.edu
* Teresa Datta tdatta@g.harvard.edu

## What's in the name "differtless"?

It's a portmanteau of "differentiation" and "effortless"

## Codecov badge

[![codecov](https://codecov.io/gh/differtless/cs107-FinalProject/branch/master/graph/badge.svg?token=AN6QT71WV9)](https://codecov.io/gh/differtless/cs107-FinalProject)

## Travis CI badge

[![Build Status](https://travis-ci.com/differtless/cs107-FinalProject.svg?token=ZrM8oyab1Y4rgKUpwoqF&branch=master)](https://travis-ci.com/differtless/cs107-FinalProject)

## Inclusion Statement 
Differtless is committed to developing a culture of inclusion. Only by learning from a diverse community of contributors are we able to build the best product and grow to be our best selves. We welcome all educational levels, cultures, ethnicities, races, gender identities and expressions, nations of origin, ages, languages spoken, veteran’s status, colors, religions, disabilities, geographic backgrounds, sexual orientations and beliefs.

*We embrace your unique background, because you belong here.*

## Broader Impact and Inclusivity Statement

One of the key applications of automatic differentiation is for optimization, and we specifically explored the implementations and utility of minimization via scipy.optimize.minimize. Although the technical details for how the minima of these functions may feel far removed from any potential negative intent, it is vital to remove our rose-colored glasses and remember that technology is never created in a void. 

Imagine, for example a machine learning algorithm for which we have determined a loss function- the most obvious way to get the best model would then be to minimize the loss function, potentially via our AD package. That's in large part because as computer scientists, our understandings of what makes a model the "best" focus accuracy metrics as optimized via a loss function. However, for models with real-world impact, they need to be both **accurate and fair**. These are two separate objectives with different metrics of success- accuracy may be measured by our loss function, but fairness requires a deeper dive:  understanding the pre-existing biases in our dataset, comparing the accuracy/false positive/false negative rates between any protected groups. 
When accuracy is too heavily optimized, the real-world impacts can be grave. The effects of algorithmic bias have caused it to be harder for certain minority groups to be hired under the same qualifications (2), and in the contexts of facial recognition systems, at the expense of ethical introspection, these algorithms are known to have lower accuracy on people of color and have caused the unnecessary and traumatic arrests of innocent men of color in the U.S. (4) and the tracking and controlling of the Uighur minority group in China (1).

However, there are still strong motivations for pursuing this type of work and potential positive impacts. Recently, in academic fields such as flow topology (5) and aerodynamic design frameworks (6), automatic differentiation have been used to advance the understanding of necessary and previouisly less understood problems.

Marginalized groups are still severely underrepresented in tech, and this disparity is even more apparent in the open source community. 
For example, a Toptal study (7) found that in their random sample, just 5.4% of GitHub users with >10 contributions are female.
To this end, we've attempted to address this subtle barriers in order to make our repo as welcoming and inclusive as possible. The first step we took to make clear that all backgrounds belong here in our contributing community was to develop an Inclusion Statement to reflect our commitment to fostering a welcoming community. This is one of the first things that users will see in our repo. 

For our software project, our code contribution is as transparent as possible. Pull requests can be approved and reviewed by any member of our team, and if there is any concern about a PR being approved, a group discussion is initiated to ensure proper vetting and open communication is maintained. 
It is also worth mentioning that our (randomly assigned) team brings a diverse array of perspectives- both from an internal diversity viewpoint (gender, ethnicity, age, etc.) but also from an academic background viewpoint. We represent a variety of fields and a range of programs under the Harvard umbrella.
This is important to note since as Anna-Chiara suggests in the Toptal study, projects that show diversity in their leadership promote a culture of inclusivity within its contributors. 
We recognize, however, that there are still barriers to certain groups that we have not yet accounted for. One of these is how non-native English speakers will interact with our code. Our entire documentation and examples are written in English, and so if given more time, we would want to make sure there is access to accurately translated versions of our documentation (or at least that it is Google Translate friendly).



1. https://www.nytimes.com/2019/04/14/technology/china-surveillance-artificial-intelligence-racial-profiling.html
2. https://www.vox.com/recode/2020/1/1/21043000/artificial-intelligence-job-applications-illinios-video-interivew-act
3. https://www.theverge.com/2019/11/11/20958953/apple-credit-card-gender-discrimination-algorithms-black-box-investigation
4. https://www.nytimes.com/2020/06/24/technology/facial-recognition-arrest.html
5. https://link.springer.com/article/10.1007/s00158-017-1708-2
6. https://www.ercoftac.org/downloads/bulletin-docs/ercoftac_bulletin_102.pdf 
7. https://www.toptal.com/open-source/is-open-source-open-to-women 
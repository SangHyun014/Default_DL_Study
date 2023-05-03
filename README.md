# RestApp

<div align="center">
<img width="329" alt="image" src="https://github.com/kwarkmc/RestApp/blob/3ef8ce603e59da37405c2a701fc839226da04a65/Documents/pic/Logo.png">

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkwarkmc%2FRestApp&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

# RestApp
> **한양대학교 ERICA 전자공학부 전공학회 DEFAULT 딥러닝 세미나 결과물** <br/> **개발기간: 2022.07 ~ 2022.09**

## 세미나 멤버 소개

|      배성현       |          주상현         |       곽민창         |                                                                                                               
| :------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | 
|   <img width="160px" src="https://avatars.githubusercontent.com/u/94032326?v=4" />    |                      <img width="160px" src="https://avatars.githubusercontent.com/u/87685922?v=4" />    |                   <img width="160px" src="https://avatars.githubusercontent.com/u/41298500?v=4"/>   |
|   [@hunction](https://github.com/hunction)   |    [@SangHyun014](https://github.com/SangHyun014)  | [@kwarkmc](https://github.com/kwarkmc)  |
| 한양대학교 ERICA 전자공학부      4학년 | 한양대학교 ERICA 전자공학부     4학년 | 한양대학교 ERICA 전자공학부     3학년 |

## 딥러닝 세미나 소개

**한양대학교 ERICA 전자공학부 전공학회 DEFAULT** 에서 학기 중에 진행된 딥러닝 스터디를 바탕으로 **더욱 심화된 DL/ML** 관련 기술 및 논문에 대해 리뷰하고 실제 모델에 적용하여 개발 경험을 쌓고자 하는 목적으로 시작된 학생 모임이다.

## 프로젝트 소개

에서의 첫 번째 개발 프로젝트로, 

## 시작 가이드
### Requirements
For building and running the application you need:

- [Node.js 14.19.3](https://nodejs.org/ca/blog/release/v14.19.3/)
- [Npm 9.2.0](https://www.npmjs.com/package/npm/v/9.2.0)
- [Strapi 3.6.6](https://www.npmjs.com/package/strapi/v/3.6.6)

### Installation
``` bash
$ git clone https://github.com/Voluntain-SKKU/Voluntain-2nd.git
$ cd Voluntain-2nd
```
#### Backend
```
$ cd strapi-backend
$ nvm use v.14.19.3
$ npm install
$ npm run develop
```

#### Frontend
```
$ cd voluntain-app
$ nvm use v.14.19.3
$ npm install 
$ npm run dev
```

---

## Stacks 🐈

### Environment
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)             

### Config
![npm](https://img.shields.io/badge/npm-CB3837?style=for-the-badge&logo=npm&logoColor=white)        

### Development
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=Javascript&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Strapi](https://img.shields.io/badge/Strapi-2F2E8B?style=for-the-badge&logo=Strapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=Next.js&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?style=for-the-badge&logo=Bootstrap&logoColor=white)
![Material UI](https://img.shields.io/badge/Material%20UI-007FFF?style=for-the-badge&logo=MUI&logoColor=white)

### Communication
![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=Slack&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=Notion&logoColor=white)
![GoogleMeet](https://img.shields.io/badge/GoogleMeet-00897B?style=for-the-badge&logo=Google%20Meet&logoColor=white)

---
## 화면 구성 📺
| 메인 페이지  |  소개 페이지   |
| :-------------------------------------------: | :------------: |
|  <img width="329" src="https://user-images.githubusercontent.com/50205887/208036155-a57900f7-c68a-470d-923c-ff3c296ea635.png"/> |  <img width="329" src="https://user-images.githubusercontent.com/50205887/208036645-a76cf400-85bc-4fa2-af72-86d2abf61366.png"/>|  
| 강좌 소개 페이지   |  강의 영상 페이지   |  
| <img width="329" src="https://user-images.githubusercontent.com/50205887/208038737-2b32b7d2-25f4-4949-baf5-83b5c02915a3.png"/>   |  <img width="329" src="https://user-images.githubusercontent.com/50205887/208038965-43a6318a-7b05-44bb-97c8-b08b0495fba7.png"/>     |

---
## 주요 기능 📦

### ⭐️ 강좌 선택 및 강의 영상 시청 기능
- Scratch, Python 2개 강좌 및 각 강좌마다 10개 가량의 강의 영상 제공
- 추후 지속적으로 강좌 추가 및 업로드 예정

### ⭐️ 강의 관련 및 단체에 대한 자유로운 댓글 작성 가능
- Disqus를 이용하여 강의 관련 질문이나 단체에 대한 질문 작성 가능

### ⭐️ 이어 학습하기 기능
- Cookie 기능을 이용하여 이전에 학습했던 내용 이후부터 바로 학습 가능
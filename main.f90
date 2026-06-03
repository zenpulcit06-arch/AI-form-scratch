program main
    use, intrinsic :: iso_fortran_env
    use network
    use Readbin
    use sgd
    use SaveingData

    implicit none
    integer(int64) :: sample_size, n_feature, classes, N, n_epoch, i, batch_size,mode
    integer(int32) :: height, width, n_size, garbage
    character(len = 1000) :: filename, filename2, savefile_name
    real(real64) :: start, finish,loss, learning_rate, acc, lamda
    real(real64), allocatable :: x(:,:) , y(:,:)
    type(layer), allocatable :: net(:)
    integer(int64), allocatable :: neurons(:)

    mode = 0
    print *, 'Enter 1 to load model, 2 to load from pytorch and anything else integer to train model'
    read (*,*) mode

    if ( mode .eq. 1 ) then
        print *, 'Enter file name'
        read(*,*) savefile_name
        call load_network(net,savefile_name,N)
    else if (mode .eq. 2) then
        print *, 'Enter file name'
        read(*,*) savefile_name
        call load_pytorch(net,savefile_name,N)
    else
    print *, 'Enter file name:'
    read (*,*) filename

    print *, 'opening file'
    open(unit=10, file=filename, form='unformatted', access='stream', status='old')
    print *, 'file open'

    print *, 'Enter sol filename:'
    read(*,*) filename2
    open(unit=11, file=filename2, form='unformatted', access='stream', status='old')


    print *,'classes size ='
    read(*,*) classes


    print *,'No. of layer ='
    read(*,*) N

    print *, 'Enter no. of epochs'
    read(*,*) n_epoch

    print *, 'Enter no. of batch size'
    read(*,*) batch_size

    print *, 'Enter learning rate'
    read(*,*) learning_rate

    print *, 'Enter lamda'
    read(*,*) lamda

    call read_int32(10,garbage)
    call read_int32(10,n_size)
    call read_int32(10,height)
    call read_int32(10,width)

    call read_int32(11, garbage)  
    call read_int32(11, garbage) 

    sample_size = int(n_size,int64)
    n_feature = height * width

    if (allocated(y)) deallocate(y)
    allocate(y(sample_size,classes))

    if (allocated(x)) deallocate(x)
    allocate(x(sample_size,n_feature))

    if (allocated(net)) deallocate(net)
    allocate(net(N))
    if (allocated(neurons)) deallocate(neurons)
    allocate(neurons(N))

    do i = 1, N
        print *, 'Enter no. of neurons in layer',i
        read(*,*) neurons(i)
    end do

    call intialize_layer(net(1),n_feature,neurons(1),sample_size)

    do i = 2,N
        call intialize_layer(net(i),neurons(i-1),neurons(i),sample_size)
    end do
    
    call cpu_time(start)

    call read_label(11,y)
    call read_images(10,height,width,x)
    close(10)
    close(11)

    call sgd_fit(net,x,y,learning_rate,n_epoch,sample_size,N,loss,lamda,batch_size)
    call forward_pass_network(net,x,N,sample_size)
    acc = accuracy(net(N)%H,y,sample_size)
    call cpu_time(finish)
    
    print *, 'Loss =',loss
    print *, 'Accuracy =', acc
    print *, 'Time of execution =', finish - start,'second'
    end if



    print *, 'Enter test file name:'
    read (*,*) filename

    print *, 'opening test file'
    open(unit=10, file=filename, form='unformatted', access='stream', status='old')

    print *, 'Enter test sol filename:'
    read(*,*) filename2

    print *, 'opening test sol file'
    open(unit=11, file=filename2, form='unformatted', access='stream', status='old')

    call read_int32(10,garbage)
    call read_int32(10,n_size)
    call read_int32(10,height)
    call read_int32(10,width)

    call read_int32(11, garbage)  
    call read_int32(11, garbage) 

    sample_size = int(n_size,int64)
    n_feature = height * width

    classes = size(net(N)%W,2)

    if (allocated(y)) deallocate(y)
    allocate(y(sample_size,classes))

    if (allocated(x)) deallocate(x)
    allocate(x(sample_size,n_feature))

    call read_label(11,y)
    call read_images(10,height,width,x)
    close(10)
    close(11)
    
    do i = 1, N
        call resize_H(net(i), sample_size)
    end do
    
    call forward_pass_network(net,x,N,sample_size)
    acc = accuracy(net(N)%H,y,sample_size)
    
    print *, 'test Accuracy =', acc

    mode = 0
    print *,'Enter 1 to save file, 2 to save in pytorch format, and any other integer to continue'
    read(*,*) mode

    if ( mode .eq. 1 ) then
        print *,'Enter file name'
        read(*,*) savefile_name
        call save_network(net,savefile_name)
    else if (mode .eq. 2) then
        print *,'Enter file name'
        read(*,*) savefile_name
        call save_pytorch(net,savefile_name)
    end if

end program main